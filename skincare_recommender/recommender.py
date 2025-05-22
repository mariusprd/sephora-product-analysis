import pandas as pd
import numpy as np
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import torch # Ensure torch is imported for device checks
import os 
import concurrent.futures # For multithreading

# --- NLTK Setup ---
try:
    english_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    word_tokenize("test") 
except LookupError:
    print("NLTK resources not found. Downloading for recommender_v2...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    english_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

def preprocess_prompt_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    filtered_tokens = [token for token in lemmatized_tokens if token.isalnum()]
    return " ".join(filtered_tokens)

class SkincareRecommenderV2:
    def __init__(self, product_profiles_path, segments_data_path, sentence_model_name='all-MiniLM-L6-v2', num_threads=os.cpu_count()):
        self.product_profiles_path = product_profiles_path
        self.segments_data_path = segments_data_path
        self.sentence_model_name = sentence_model_name
        
        # Enhanced device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # Optional: Clear MPS cache if encountering issues, use with caution
            # torch.mps.empty_cache() 
        else:
            self.device = torch.device("cpu")
        
        self.num_threads = num_threads if num_threads else os.cpu_count()

        print(f"--- Initializing SkincareRecommenderV2 (using device: {self.device}, threads: {self.num_threads}) ---")
        print("INFO: Product pros/cons explanations are derived from pre-aggregated review texts.")
        print("      The recommender does not translate this text. Ensure source data is in the desired language (English).")
        
        self._load_product_profiles()
        self._load_client_segments()
        
        print(f"Loading sentence transformer model for prompt embedding: {self.sentence_model_name}...")
        model_load_start_time = time.time()
        try:
            # Pass the selected device to the SentenceTransformer
            self.sbert_model = SentenceTransformer(self.sentence_model_name, device=self.device)
        except Exception as e:
            print(f"ERROR: Could not load Sentence Transformer model '{self.sentence_model_name}': {e}")
            raise
        print(f"Sentence Transformer model loaded in {time.time() - model_load_start_time:.2f}s.")
        print("Recommender V2 initialized successfully.")

    def _load_product_profiles(self):
        print(f"Loading pre-calculated product profiles from: {self.product_profiles_path}")
        if not os.path.exists(self.product_profiles_path):
            print(f"ERROR: Product profiles file not found at {self.product_profiles_path}.")
            raise FileNotFoundError(f"Product profiles file not found: {self.product_profiles_path}")
        try:
            self.product_profiles_df = pd.read_parquet(self.product_profiles_path)
            self.product_profiles_df['pro_embedding'] = self.product_profiles_df['pro_embedding'].apply(np.array)
            self.product_profiles_df['con_embedding'] = self.product_profiles_df['con_embedding'].apply(np.array)
            print(f"Loaded {len(self.product_profiles_df)} product profiles.")
        except Exception as e:
            print(f"ERROR: Could not load or parse product profiles from {self.product_profiles_path}: {e}")
            raise

    def _load_client_segments(self):
        print(f"Loading client segments data from: {self.segments_data_path}")
        if not os.path.exists(self.segments_data_path):
            print(f"ERROR: Client segments file not found at {self.segments_data_path}.")
            raise FileNotFoundError(f"Client segments file not found: {self.segments_data_path}")
        try:
            segments_df = pd.read_csv(self.segments_data_path)
            if 'author_id' not in segments_df.columns or 'cluster_id' not in segments_df.columns:
                print("ERROR: Segments CSV must contain 'author_id' and 'cluster_id'. Renaming 'customer_id' if present.")
                if 'customer_id' in segments_df.columns and 'author_id' not in segments_df.columns:
                    segments_df.rename(columns={'customer_id': 'author_id'}, inplace=True)
                else:
                    raise ValueError("Segments CSV missing required 'author_id' or 'cluster_id' columns.")
            
            segments_df['author_id'] = segments_df['author_id'].astype(str)
            self.author_to_segment = pd.Series(segments_df.cluster_id.values, index=segments_df.author_id).to_dict()
            print(f"Loaded segment data for {len(self.author_to_segment)} authors.")
        except Exception as e:
            print(f"ERROR: Could not load client segments from {self.segments_data_path}: {e}")
            raise

    def get_client_segment(self, client_id):
        return self.author_to_segment.get(str(client_id), 'UnknownSegment')

    def _calculate_single_product_score(self, args_tuple):
        """Helper function for multithreading the scoring part."""
        idx, product_profile, pro_similarity, con_similarity, target_client_group, segment_boost_factor, pro_weight, con_penalty_factor = args_tuple
        
        base_score = pro_similarity * pro_weight
        if con_similarity > 0:
            base_score -= (con_similarity * con_penalty_factor)

        segment_rating_col = f'segment_{target_client_group}_avg_rating'
        segment_review_count_col = f'segment_{target_client_group}_review_count'
        
        final_score = base_score
        # Check if segment specific columns exist in the product_profile (which is a Series)
        if segment_rating_col in product_profile.index and segment_review_count_col in product_profile.index and product_profile[segment_review_count_col] > 0:
            avg_rating_for_segment = product_profile[segment_rating_col]
            if avg_rating_for_segment > 3.5:
                 final_score *= (segment_boost_factor * (avg_rating_for_segment / 5.0))
            elif avg_rating_for_segment < 2.5 and avg_rating_for_segment > 0:
                 final_score *= (1.0 / segment_boost_factor)
        
        return {
            'product_id': product_profile['product_id'],
            'product_name': product_profile['product_name'],
            'brand_name': product_profile['brand_name'],
            'score': float(final_score),
            'explanation_pros': product_profile['positive_aggregate_text'],
            'explanation_cons': product_profile['negative_aggregate_text'],
            'segment_avg_rating_for_user': product_profile.get(segment_rating_col, "N/A"),
            'segment_review_count_for_user': product_profile.get(segment_review_count_col, 0)
        }

    def get_recommendations(self, prompt, client_id, top_n=5, segment_boost_factor=1.2, pro_weight=1.0, con_penalty_factor=0.5):
        print(f"\n--- Generating recommendations for client {client_id} (V2) ---")
        recommendation_start_time = time.time()

        client_id_str = str(client_id)
        target_client_group = self.get_client_segment(client_id_str)
        print(f"Client ID: {client_id_str}, Target Segment: {target_client_group}")

        processed_prompt = preprocess_prompt_text(prompt)
        if not processed_prompt:
            print("Warning: Empty prompt after processing. Cannot generate recommendations.")
            return []
        
        print(f"Original prompt: '{prompt}' -> Processed: '{processed_prompt}'")
        prompt_embedding_start_time = time.time()
        # Ensure prompt_embedding is created on the correct device
        prompt_embedding = self.sbert_model.encode(processed_prompt, convert_to_tensor=True)
        if prompt_embedding.device.type != self.device.type: # SBERT might default to CPU if not specified in call
             prompt_embedding = prompt_embedding.to(self.device)
        print(f"Prompt embedded in {time.time() - prompt_embedding_start_time:.2f}s (on {prompt_embedding.device}).")

        # Prepare embeddings on the target device
        all_pro_embeddings = torch.tensor(np.stack(self.product_profiles_df['pro_embedding'].values), device=self.device, dtype=torch.float)
        all_con_embeddings = torch.tensor(np.stack(self.product_profiles_df['con_embedding'].values), device=self.device, dtype=torch.float)

        if prompt_embedding.ndim == 1:
            prompt_embedding_2d = prompt_embedding.unsqueeze(0)
        else:
            prompt_embedding_2d = prompt_embedding
            
        # Cosine similarity calculation (runs on self.device)
        cos_sim_start_time = time.time()
        pro_similarities = util.pytorch_cos_sim(prompt_embedding_2d, all_pro_embeddings)[0].cpu().numpy()
        con_similarities = util.pytorch_cos_sim(prompt_embedding_2d, all_con_embeddings)[0].cpu().numpy()
        print(f"Cosine similarities calculated in {time.time() - cos_sim_start_time:.2f}s.")

        # Prepare arguments for multithreaded scoring
        tasks_args = []
        for idx, product_profile_row in self.product_profiles_df.iterrows():
            tasks_args.append((
                idx, 
                product_profile_row,
                pro_similarities[idx], 
                con_similarities[idx], 
                target_client_group, 
                segment_boost_factor, 
                pro_weight, 
                con_penalty_factor
            ))
        
        scores = []
        print(f"Calculating final scores for {len(tasks_args)} products using {self.num_threads} threads...")
        scoring_start_time = time.time()
        if self.num_threads > 1 and len(tasks_args) > 1 :
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(executor.map(self._calculate_single_product_score, tasks_args))
                scores.extend(results)
        else:
            for args_tuple in tasks_args:
                scores.append(self._calculate_single_product_score(args_tuple))

        print(f"Scoring completed in {time.time() - scoring_start_time:.2f}s.")

        sorted_recommendations = sorted(scores, key=lambda x: x['score'], reverse=True)
        
        print(f"Recommendation generation completed in {time.time() - recommendation_start_time:.2f}s.")
        return sorted_recommendations[:top_n]

