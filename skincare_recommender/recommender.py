import pandas as pd
import numpy as np
import re
import time
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize # Added sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline # For summarization
import torch 
import os 
import concurrent.futures 
import math 

# --- NLTK Setup ---
try:
    english_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    nltk.data.find('tokenizers/punkt') # Ensure punkt is available for sent_tokenize
except LookupError:
    print("NLTK resources (punkt, stopwords, wordnet, omw-1.4) not found. Downloading for recommender_v2...")
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
    def __init__(self, product_profiles_path, segments_data_path, 
                 sentence_model_name='all-MiniLM-L6-v2', 
                 summarizer_model_name='sshleifer/distilbart-cnn-12-6',
                 num_threads=None):
        self.product_profiles_path = product_profiles_path
        self.segments_data_path = segments_data_path
        self.sentence_model_name = sentence_model_name
        self.summarizer_model_name = summarizer_model_name
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.summarizer_pipeline_device_arg = 0 
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            self.summarizer_pipeline_device_arg = -1 
            print(f"INFO: SBERT model will use MPS device ({self.device}). Summarizer pipeline device argument set to CPU (-1) for stability.")
        else:
            self.device = torch.device("cpu")
            self.summarizer_pipeline_device_arg = -1 
        
        self.num_threads = num_threads if num_threads else os.cpu_count()

        print(f"--- Initializing SkincareRecommenderV2 (SBERT device: {self.device}, Summarizer pipeline device arg: {self.summarizer_pipeline_device_arg}, threads: {self.num_threads}) ---")
        
        self._load_product_profiles()
        self._load_client_segments() 
        
        print(f"Loading SBERT model for prompt embedding: {self.sentence_model_name}...")
        sbert_load_start_time = time.time()
        try:
            self.sbert_model = SentenceTransformer(self.sentence_model_name, device=self.device)
        except Exception as e:
            print(f"ERROR: Could not load SBERT model '{self.sentence_model_name}': {e}")
            raise
        print(f"SBERT model loaded in {time.time() - sbert_load_start_time:.2f}s.")

        print(f"Loading Summarization model: {self.summarizer_model_name}...")
        summarizer_load_start_time = time.time()
        try:
            self.summarizer = pipeline("summarization", model=self.summarizer_model_name, tokenizer=self.summarizer_model_name, device=self.summarizer_pipeline_device_arg)
            self.actual_summarizer_max_input_tokens = self.summarizer.tokenizer.model_max_length if hasattr(self.summarizer.tokenizer, 'model_max_length') else 512 
        except Exception as e:
            print(f"ERROR: Could not load Summarizer model '{self.summarizer_model_name}': {e}")
            self.summarizer = None
            self.actual_summarizer_max_input_tokens = 0
            print("WARNING: Summarizer model failed to load. Personalized summaries will not be available.")
        print(f"Summarizer model loaded in {time.time() - summarizer_load_start_time:.2f}s. Max input tokens for summarizer: {self.actual_summarizer_max_input_tokens}")
        
        print("Recommender V2 initialized successfully.")

    def _load_product_profiles(self):
        print(f"Loading product profiles from: {self.product_profiles_path}")
        if not os.path.exists(self.product_profiles_path):
            print(f"ERROR: Product profiles file not found: {self.product_profiles_path}.")
            raise FileNotFoundError(f"Product profiles file not found: {self.product_profiles_path}")
        try:
            self.product_profiles_df = pd.read_parquet(self.product_profiles_path)
            for col in ['pro_embedding', 'con_embedding']:
                if col not in self.product_profiles_df.columns:
                    raise ValueError(f"Missing required embedding column '{col}' in product profiles.")
                self.product_profiles_df[col] = self.product_profiles_df[col].apply(np.array)
            for col in ['positive_aggregate_text', 'negative_aggregate_text', 'total_recent_reviews']:
                if col not in self.product_profiles_df.columns:
                     raise ValueError(f"Missing required column '{col}'. Rerun preprocessing (V3+).")
            print(f"Loaded {len(self.product_profiles_df)} product profiles.")
        except Exception as e:
            print(f"ERROR loading product profiles from {self.product_profiles_path}: {e}")
            raise

    def _load_client_segments(self):
        print(f"Loading client segments from: {self.segments_data_path}")
        if not os.path.exists(self.segments_data_path):
            print(f"ERROR: Client segments file not found: {self.segments_data_path}.")
            raise FileNotFoundError(f"Client segments file not found: {self.segments_data_path}")
        try:
            segments_df = pd.read_csv(self.segments_data_path)
            if 'cluster_id' in segments_df.columns: 
                segments_df.rename(columns={'cluster_id': 'client_group'}, inplace=True)
            elif 'client_group' not in segments_df.columns:
                if 'customer_id' in segments_df.columns and 'author_id' not in segments_df.columns: 
                     segments_df.rename(columns={'customer_id': 'author_id'}, inplace=True)
                else:
                    raise ValueError("Segments CSV must contain 'author_id' and ('client_group' or 'cluster_id').")

            if 'author_id' not in segments_df.columns or 'client_group' not in segments_df.columns:
                 raise ValueError("Critical error: 'author_id' or 'client_group' (after renames) not found.")

            segments_df['author_id'] = segments_df['author_id'].astype(str)
            segments_df['client_group'] = segments_df['client_group'].astype(str) 
            self.author_to_segment = pd.Series(segments_df.client_group.values, index=segments_df.author_id).to_dict()
        except Exception as e:
            print(f"ERROR loading client segments from {self.segments_data_path}: {e}")
            raise

    def get_client_segment(self, client_id):
        return self.author_to_segment.get(str(client_id), 'UnknownSegment')

    def _extract_key_sentences(self, text, num_sentences=2, max_chars=160): # Increased max_chars slightly
        """A simple fallback to extract first few sentences, limited by characters."""
        if not text or not isinstance(text, str) or text.strip().lower() in ["no positive reviews", "no negative reviews"]:
            return f"No specific review content available."
        try:
            sentences = sent_tokenize(text) 
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3] # Min 4 words
            
            if not meaningful_sentences: # If no meaningful sentences found
                 return text[:max_chars].strip(' .') + ("..." if len(text) > max_chars else "")

            extracted_text = ". ".join(meaningful_sentences[:num_sentences])
            if not extracted_text.endswith('.'): # Ensure it ends with a period if not empty
                extracted_text += "."
                
            if not extracted_text.strip() or extracted_text == ".":
                 return text[:max_chars].strip(' .') + ("..." if len(text) > max_chars else "")
            
            if len(extracted_text) > max_chars:
                last_space = extracted_text.rfind(' ', 0, max_chars)
                if last_space != -1:
                    extracted_text = extracted_text[:last_space] + "..."
                else: 
                    extracted_text = extracted_text[:max_chars-3] + "..."
            return extracted_text.strip(' .')

        except Exception as e_sent: 
            print(f"    INFO: sent_tokenize failed for snippet extraction, using char slice. Error: {e_sent}")
            return text[:max_chars].strip(' .') + ("..." if len(text) > max_chars else "")

    def _generate_personalized_summary(self, user_prompt, product_name, aggregate_review_text, summary_type, target_client_group):
        if not self.summarizer: 
            return f"Key review points: {self._extract_key_sentences(aggregate_review_text)}" # Use fallback if no summarizer
            
        placeholder_text_positive = "no positive reviews"
        placeholder_text_negative = "no negative reviews"
        current_placeholder = placeholder_text_positive if summary_type == "pros" else placeholder_text_negative
        
        # print(f"    DEBUG RAW AGG TEXT for \"{product_name}\" ({summary_type}): '{str(aggregate_review_text)[:300]}...'")

        if not aggregate_review_text or not isinstance(aggregate_review_text, str):
            return f"No specific {summary_type} review content for {product_name} (data error)."
        
        cleaned_aggregate_text = aggregate_review_text.strip().lower()
        if cleaned_aggregate_text == current_placeholder:
            return f"No specific {summary_type} review content identified for {product_name}."

        meaningful_text_word_count = len(cleaned_aggregate_text.split())
        if meaningful_text_word_count < 20: # If input is too short, fallback to sentence extraction
             return f"Not enough distinct review content for {summary_type} ({meaningful_text_word_count} words). Key points: {self._extract_key_sentences(aggregate_review_text)}"

        input_for_summarizer = aggregate_review_text 
        
        try:
            # print(f"    DEBUG FINAL INPUT to summarizer ({summary_type}) for \"{product_name}\" (first 300 chars): '{input_for_summarizer[:300]}...'")
            summary_list = self.summarizer(
                [input_for_summarizer], 
                max_length=60,  # Increased max_length
                min_length=15,  # Increased min_length
                do_sample=False, 
                truncation=True 
            )
            # print(f"    DEBUG SUMMARIZER RAW OUTPUT LIST for \"{product_name}\" ({summary_type}): {summary_list}")

            if summary_list and isinstance(summary_list, list) and len(summary_list) > 0 and \
               isinstance(summary_list[0], dict) and 'summary_text' in summary_list[0] and \
               summary_list[0]['summary_text'] and summary_list[0]['summary_text'].strip():
                
                generated_text = summary_list[0]['summary_text'].strip()
                
                # Check if the summary is too short or if it's just copying a large part of the input
                first_few_words_of_input = " ".join(aggregate_review_text.split()[:15])
                # If summary is shorter than (min_length / 2) or too similar to input start
                if len(generated_text.split()) < (15 // 2) or \
                   generated_text.lower().startswith(first_few_words_of_input.lower()[:max(15, len(generated_text)-10)]): 
                    # print(f"    WARNING: Summarizer output for \"{product_name}\" ({summary_type}) too short or extractive: '{generated_text}' -> Using fallback.")
                    return f"Key review points: {self._extract_key_sentences(aggregate_review_text)}"
                
                return generated_text
            else: 
                return f"Key review points: {self._extract_key_sentences(aggregate_review_text)}" # Fallback
        except Exception as e:
            print(f"    ERROR during personalized summarization for \"{product_name}\" ({summary_type}): {e}")
            return f"Error summarizing. Key review points: {self._extract_key_sentences(aggregate_review_text)}"


    def _calculate_single_product_score_and_info(self, args_tuple):
        # (No changes to this method from the previous version you confirmed was better for scoring)
        idx, product_profile, pro_similarity, con_similarity, target_client_group, segment_boost_factor, pro_weight, con_penalty_factor, review_count_config = args_tuple
        
        base_score = pro_similarity * pro_weight
        if con_similarity > 0.15: 
            base_score -= (con_similarity * con_penalty_factor)

        final_score = base_score 
        
        segment_rating_col = f'segment_{target_client_group}_avg_rating'
        segment_review_count_col = f'segment_{target_client_group}_review_count'
        segment_reviews_for_this_product = product_profile.get(segment_review_count_col, 0)

        if segment_rating_col in product_profile.index and \
           segment_review_count_col in product_profile.index and \
           segment_reviews_for_this_product >= review_count_config['segment_min_reviews']:
            avg_rating_for_segment = product_profile[segment_rating_col]
            if avg_rating_for_segment > review_count_config.get('segment_good_rating_threshold', 3.9): 
                 final_score *= (segment_boost_factor * (avg_rating_for_segment / 5.0)) 
            elif avg_rating_for_segment < review_count_config.get('segment_poor_rating_threshold', 2.7) and avg_rating_for_segment > 0:
                 final_score *= (1.0 / segment_boost_factor)
        elif segment_reviews_for_this_product == 0 and review_count_config.get('penalize_zero_segment_reviews', True):
            final_score *= review_count_config.get('zero_segment_review_multiplier', 0.3) 


        review_count = product_profile['total_recent_reviews']
        
        if review_count < review_count_config['absolute_min_reviews']: 
            final_score *= review_count_config.get('absolute_min_review_multiplier', 0.01) 
        elif review_count < review_count_config['hard_min_review_threshold']: 
            final_score *= review_count_config.get('hard_min_review_multiplier', 0.2) 
        elif review_count < review_count_config['min_threshold_for_penalty']: 
            penalty_factor = (review_count_config['min_threshold_for_penalty'] - review_count) / float(review_count_config['min_threshold_for_penalty']) 
            final_score *= (1 - (review_count_config['penalty_strength'] * penalty_factor) )
        elif review_count > review_count_config['min_threshold_for_bonus']:
            bonus = math.log1p(review_count - review_count_config['min_threshold_for_bonus']) * review_count_config['bonus_strength']
            final_score += min(bonus, review_count_config['max_bonus'])

        final_score = max(final_score, -2.0) 

        return {
            'product_id': product_profile['product_id'],
            'product_name': product_profile['product_name'],
            'brand_name': product_profile['brand_name'],
            'score': float(final_score),
            'positive_aggregate_text': product_profile['positive_aggregate_text'], 
            'negative_aggregate_text': product_profile['negative_aggregate_text'], 
            'total_recent_reviews': review_count,
            'segment_avg_rating_for_user': product_profile.get(segment_rating_col, "N/A"),
            'segment_review_count_for_user': segment_reviews_for_this_product
        }


    def get_recommendations(self, prompt, client_id, top_n=5, 
                            segment_boost_factor=1.3, 
                            pro_weight=1.0, 
                            con_penalty_factor=0.75, 
                            review_count_config=None ):
        
        if review_count_config is None:
            review_count_config = {
                'absolute_min_reviews': 3,        
                'absolute_min_review_multiplier': 0.01, 
                'hard_min_review_threshold': 7,  
                'hard_min_review_multiplier': 0.2, 
                'min_threshold_for_penalty': 20,  
                'penalty_strength': 0.6,          
                'min_threshold_for_bonus': 50,    
                'bonus_strength': 0.1,            
                'max_bonus': 0.35,                
                'segment_min_reviews': 2,         
                'segment_good_rating_threshold': 3.8, 
                'segment_poor_rating_threshold': 2.8, 
                'penalize_zero_segment_reviews': True, 
                'zero_segment_review_multiplier': 0.3 
            }

        print(f"\n--- Generating recommendations for client {client_id} (V2 - Online Personalized Summaries, RevScore V6) ---")
        overall_recommendation_start_time = time.time()

        client_id_str = str(client_id)
        target_client_group = self.get_client_segment(client_id_str) 
        print(f"Client ID: {client_id_str}, Target Segment (cluster_id): {target_client_group}")

        processed_prompt = preprocess_prompt_text(prompt)
        if not processed_prompt:
            print("Warning: Empty prompt. Cannot generate recommendations.")
            return []
        
        print(f"Original prompt: '{prompt}' -> Processed: '{processed_prompt}'")
        prompt_embedding_start_time = time.time()
        prompt_embedding = self.sbert_model.encode(processed_prompt, convert_to_tensor=True)
        if prompt_embedding.device.type != self.device.type:
             prompt_embedding = prompt_embedding.to(self.device)
        print(f"Prompt embedded in {time.time() - prompt_embedding_start_time:.2f}s (on {prompt_embedding.device}).")

        all_pro_embeddings = torch.tensor(np.stack(self.product_profiles_df['pro_embedding'].values), device=self.device, dtype=torch.float)
        all_con_embeddings = torch.tensor(np.stack(self.product_profiles_df['con_embedding'].values), device=self.device, dtype=torch.float)

        if prompt_embedding.ndim == 1:
            prompt_embedding_2d = prompt_embedding.unsqueeze(0)
        else:
            prompt_embedding_2d = prompt_embedding
            
        cos_sim_start_time = time.time()
        pro_similarities = util.pytorch_cos_sim(prompt_embedding_2d, all_pro_embeddings)[0].cpu().numpy()
        con_similarities = util.pytorch_cos_sim(prompt_embedding_2d, all_con_embeddings)[0].cpu().numpy()
        print(f"Cosine similarities calculated in {time.time() - cos_sim_start_time:.2f}s.")
        
        tasks_args = []
        for idx, product_profile_row in self.product_profiles_df.iterrows():
            tasks_args.append((
                idx, product_profile_row, 
                pro_similarities[idx], con_similarities[idx], 
                target_client_group, segment_boost_factor, 
                pro_weight, con_penalty_factor, review_count_config
            ))
        
        scored_products_info = []
        # print(f"Calculating initial scores for {len(tasks_args)} products using {self.num_threads} threads...")
        scoring_start_time = time.time()
        if self.num_threads > 1 and len(tasks_args) > 1 :
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(executor.map(self._calculate_single_product_score_and_info, tasks_args))
                scored_products_info.extend(results)
        else: 
            for args_tuple in tasks_args:
                scored_products_info.append(self._calculate_single_product_score_and_info(args_tuple))
        # print(f"Initial scoring completed in {time.time() - scoring_start_time:.2f}s.")
        
        sorted_candidates = sorted(scored_products_info, key=lambda x: x['score'], reverse=True)
        
        meaningful_candidates = [cand for cand in sorted_candidates if cand['score'] > 0.01] 
        
        num_to_summarize = min(len(meaningful_candidates), top_n) 
        top_candidates_for_summary = meaningful_candidates[:num_to_summarize]

        final_recommendations = []
        if self.summarizer and top_candidates_for_summary: 
            print(f"Generating personalized summaries for top {len(top_candidates_for_summary)} candidates...")
            summarization_loop_start_time = time.time()
            
            summary_tasks_args_list = []
            for candidate_info in top_candidates_for_summary:
                summary_tasks_args_list.append({
                    'args': (processed_prompt, candidate_info['product_name'], candidate_info['positive_aggregate_text'], "pros", target_client_group),
                    'candidate_info': candidate_info, 'type': 'pros'
                })
                summary_tasks_args_list.append({
                    'args': (processed_prompt, candidate_info['product_name'], candidate_info['negative_aggregate_text'], "cons", target_client_group),
                    'candidate_info': candidate_info, 'type': 'cons'
                })

            temp_summaries = {} 
            candidate_map = {cand['product_id']: cand for cand in top_candidates_for_summary}

            if self.num_threads > 1 and len(summary_tasks_args_list) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    future_to_task = {
                        executor.submit(self._generate_personalized_summary, *task['args']): task 
                        for task in summary_tasks_args_list
                    }
                    for future in concurrent.futures.as_completed(future_to_task):
                        task_info = future_to_task[future]
                        original_candidate_info = task_info['candidate_info']
                        summary_type = task_info['type']
                        try:
                            summary_text = future.result()
                            if original_candidate_info['product_id'] not in temp_summaries:
                                temp_summaries[original_candidate_info['product_id']] = {}
                            temp_summaries[original_candidate_info['product_id']][summary_type] = summary_text
                        except Exception as exc:
                            print(f"Product {original_candidate_info['product_name']} ({summary_type}) summary generated an exception: {exc}")
                            if original_candidate_info['product_id'] not in temp_summaries:
                                temp_summaries[original_candidate_info['product_id']] = {}
                            temp_summaries[original_candidate_info['product_id']][summary_type] = f"Error during {summary_type} summary."
            else: 
                for task in summary_tasks_args_list:
                    summary_text = self._generate_personalized_summary(*task['args'])
                    original_candidate_info = task['candidate_info']
                    summary_type = task['type']
                    if original_candidate_info['product_id'] not in temp_summaries: # Ensure dict exists
                        temp_summaries[original_candidate_info['product_id']] = {}
                    temp_summaries[original_candidate_info['product_id']][summary_type] = summary_text
            
            for candidate_info in top_candidates_for_summary:
                product_id = candidate_info['product_id']
                candidate_info['pros_summary_personalized'] = temp_summaries.get(product_id, {}).get("pros", "Pros summary unavailable.")
                candidate_info['cons_summary_personalized'] = temp_summaries.get(product_id, {}).get("cons", "Cons summary unavailable.")
                candidate_info.pop('positive_aggregate_text', None)
                candidate_info.pop('negative_aggregate_text', None)
                final_recommendations.append(candidate_info)

            print(f"Personalized summarization loop took {time.time() - summarization_loop_start_time:.2f}s.")

        elif not top_candidates_for_summary:
            print("No candidates with positive scores to summarize.")
        else: 
            print("WARNING: Summarizer not available. Skipping personalized summaries.")
            for candidate_info in top_candidates_for_summary: 
                 candidate_info['pros_summary_personalized'] = "Summary not available (summarizer error)."
                 candidate_info['cons_summary_personalized'] = "Summary not available (summarizer error)."
                 candidate_info.pop('positive_aggregate_text', None)
                 candidate_info.pop('negative_aggregate_text', None)
                 final_recommendations.append(candidate_info)

        final_recommendations = sorted(final_recommendations, key=lambda x: x['score'], reverse=True)

        print(f"Overall recommendation generation completed in {time.time() - overall_recommendation_start_time:.2f}s.")
        return final_recommendations 

