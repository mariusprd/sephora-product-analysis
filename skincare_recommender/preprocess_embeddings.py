import os
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import concurrent.futures
from tqdm import tqdm
from pandarallel import pandarallel
import traceback 

# --- Configuration ---
REVIEWS_FILE_PATH = 'data/processed/reviews.csv'
SEGMENTS_FILE_PATH = 'notebooks/customer_segments_refined.csv'
OUTPUT_PROFILES_PATH = 'data/processed/product_profile_embeddings_v4.parquet'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_AGGREGATE_TEXT_LEN_FOR_EMBEDDING = 3000

# --- NLTK Setup & Globals for Parallel Processing ---
try:
    english_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    word_tokenize("test")
except LookupError:
    print("NLTK resources not found. Downloading...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    english_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

def preprocess_text_for_aggregation(text):
    """Preprocesses text for aggregation (used by pandarallel)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text) # Uses global NLTK resources
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens] # Uses global NLTK resources
    filtered_tokens = [token for token in lemmatized_tokens if token.isalnum()]
    return " ".join(filtered_tokens)

# --- Worker Function for Multiprocessing Product Profiles ---
def process_single_product(args):
    """
    Processes a single product's reviews to generate its profile.
    This function is designed to be run in a separate process.
    """
    from sentence_transformers import SentenceTransformer

    product_info_series, product_id, product_specific_reviews_df, \
    original_segment_names, model_name, device_str, max_text_len, \
    embedding_dim_val = args
    
    sbert_model_worker = SentenceTransformer(model_name, device=device_str)

    if product_specific_reviews_df.empty:
        return None

    product_specific_reviews_df['user_rating'] = pd.to_numeric(product_specific_reviews_df['user_rating'], errors='coerce')
    
    valid_ratings_df = product_specific_reviews_df.dropna(subset=['user_rating']).copy()

    good_reviews_text = " ".join(valid_ratings_df[valid_ratings_df['user_rating'] >= 4]['processed_text'].tolist())
    bad_reviews_text = " ".join(valid_ratings_df[valid_ratings_df['user_rating'] <= 2]['processed_text'].tolist())

    positive_agg_text_for_embed = good_reviews_text[:max_text_len] if good_reviews_text else "no positive reviews"
    negative_agg_text_for_embed = bad_reviews_text[:max_text_len] if bad_reviews_text else "no negative reviews"

    pro_embedding = sbert_model_worker.encode(positive_agg_text_for_embed).tolist() \
        if positive_agg_text_for_embed != "no positive reviews" else np.zeros(embedding_dim_val).tolist()
    con_embedding = sbert_model_worker.encode(negative_agg_text_for_embed).tolist() \
        if negative_agg_text_for_embed != "no negative reviews" else np.zeros(embedding_dim_val).tolist()

    total_recent_reviews_count = len(product_specific_reviews_df)

    segment_stats = {}
    all_relevant_segments = set(original_segment_names) | set(product_specific_reviews_df['cluster_id'].unique())
    
    for segment_name in all_relevant_segments:
        if segment_name is np.nan or pd.isna(segment_name):
            continue
            
        segment_reviews_for_rating = valid_ratings_df[valid_ratings_df['cluster_id'] == segment_name]
        segment_reviews_for_count = product_specific_reviews_df[product_specific_reviews_df['cluster_id'] == segment_name]

        if not segment_reviews_for_rating.empty:
            segment_stats[f'segment_{segment_name}_avg_rating'] = segment_reviews_for_rating['user_rating'].mean()
        else:
            segment_stats[f'segment_{segment_name}_avg_rating'] = 0.0

        segment_stats[f'segment_{segment_name}_review_count'] = len(segment_reviews_for_count)


    profile = {
        'product_id': product_id,
        'product_name': product_info_series['product_name'],
        'brand_name': product_info_series['brand_name'],
        'total_recent_reviews': total_recent_reviews_count,
        'pro_embedding': pro_embedding,
        'con_embedding': con_embedding,
        'positive_aggregate_text': positive_agg_text_for_embed,
        'negative_aggregate_text': negative_agg_text_for_embed,
        **segment_stats
    }
    return profile

def main():
    print("--- Starting Offline Product Profile Preprocessing (V4 - Parallel & Robust) ---")
    start_script_time = time.time()
    
    from sentence_transformers import SentenceTransformer

    # --- Determine Device ---
    if torch.cuda.is_available():
        torch_device_str = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch_device_str = "mps"
    else:
        torch_device_str = "cpu"
    print(f"Torch device for SBERT: {torch_device_str}")

    # --- Step 1: Ensure Model is Cached (Online Mode for this step only) ---
    print(f"Ensuring sentence transformer model '{SENTENCE_MODEL_NAME}' is cached locally...")
    original_hf_offline_status = os.environ.pop('HF_HUB_OFFLINE', None)
    if 'HF_HUB_OFFLINE' in os.environ:
        del os.environ['HF_HUB_OFFLINE']

    embedding_dim = 0 
    try:
        sbert_temp_for_caching = SentenceTransformer(SENTENCE_MODEL_NAME, device=torch_device_str)
        embedding_dim = sbert_temp_for_caching.get_sentence_embedding_dimension()
        del sbert_temp_for_caching 
        print(f"Model '{SENTENCE_MODEL_NAME}' (dim: {embedding_dim}) is available/cached.")
    except Exception as e:
        print(f"FATAL: Could not download/cache model '{SENTENCE_MODEL_NAME}'. Exiting. Error: {e}")
        traceback.print_exc()
        return
    finally:
        if original_hf_offline_status is not None:
            os.environ['HF_HUB_OFFLINE'] = original_hf_offline_status
        elif 'HF_HUB_OFFLINE' in os.environ:
             del os.environ['HF_HUB_OFFLINE']


    # --- Step 2: Set to OFFLINE mode for all subsequent Hugging Face Hub interactions ---
    print("Setting Hugging Face Hub to OFFLINE mode for worker processes.")
    os.environ['HF_HUB_OFFLINE'] = '1'

    # --- Load Data ---
    print(f"Loading reviews from {REVIEWS_FILE_PATH}...")
    try:
        dtype_spec = {'user_rating': str, 'author_id': str}
        print("Attempting to load CSV with Python engine for robustness...")
        reviews_df = pd.read_csv(REVIEWS_FILE_PATH, dtype=dtype_spec, engine='python')

        # Robustly convert user_rating to numeric
        reviews_df['user_rating'] = pd.to_numeric(reviews_df['user_rating'], errors='coerce')
        
        initial_review_count = len(reviews_df)
        reviews_df.dropna(subset=['user_rating'], inplace=True)
        print(f"Dropped {initial_review_count - len(reviews_df)} reviews due to non-numeric user_rating.")

    except FileNotFoundError:
        print(f"ERROR: Reviews file not found at {REVIEWS_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during reviews data loading: {e}")
        traceback.print_exc()
        return

    required_review_cols = ['author_id', 'product_id', 'product_name', 'brand_name', 'user_rating', 'review_text', 'submission_time']
    if not all(col in reviews_df.columns for col in required_review_cols):
        missing_cols = [col for col in required_review_cols if col not in reviews_df.columns]
        print(f"ERROR: Reviews CSV must contain columns: {', '.join(required_review_cols)}. Missing: {missing_cols}")
        return

    print(f"Loading client segments from {SEGMENTS_FILE_PATH}...")
    try:
        segments_df = pd.read_csv(SEGMENTS_FILE_PATH)
        # Standardize column names and types
        if 'customer_id' in segments_df.columns and 'author_id' not in segments_df.columns:
            segments_df.rename(columns={'customer_id': 'author_id'}, inplace=True)
        
        if 'author_id' not in segments_df.columns or 'cluster_id' not in segments_df.columns:
            print("ERROR: Segments CSV must contain 'author_id' and 'cluster_id' columns after potential rename.")
            return
        
        segments_df['author_id'] = segments_df['author_id'].astype(str)
        segments_df['cluster_id'] = segments_df['cluster_id'].astype(str)

    except FileNotFoundError:
        print(f"ERROR: Segments file not found at {SEGMENTS_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during segment data loading: {e}")
        traceback.print_exc()
        return

    # --- Filter Recent Reviews ---
    print("Filtering reviews from the last 2 years...")
    try:
        reviews_df['submission_time'] = pd.to_datetime(reviews_df['submission_time'], errors='coerce')
        reviews_df.dropna(subset=['submission_time'], inplace=True)
    except Exception as e:
        print(f"Error converting 'submission_time': {e}. Check format.")
        return
        
    two_years_ago = datetime.now() - timedelta(days=4*365) # Corrected back to 2 years
    recent_reviews_df = reviews_df[reviews_df['submission_time'] >= two_years_ago].copy()
    print(f"Found {len(recent_reviews_df)} recent reviews (out of {len(reviews_df)} total valid reviews).")
    if recent_reviews_df.empty:
        print("No recent reviews found. Exiting.")
        return

    # --- Preprocess Review Text (Parallelized) ---
    print("Preprocessing review text for aggregation (using pandarallel)...")
    num_text_proc_workers = max(1, (os.cpu_count() // 2) if os.cpu_count() else 1) 
    print(f"Initializing pandarallel with {num_text_proc_workers} workers.")
    pandarallel.initialize(nb_workers=num_text_proc_workers, progress_bar=True, verbose=0)
    recent_reviews_df['processed_text'] = recent_reviews_df['review_text'].parallel_apply(preprocess_text_for_aggregation)

    # --- Merge with Segments ---
    print("Merging reviews with segment data...")
    merged_reviews_df = pd.merge(recent_reviews_df, segments_df, on='author_id', how='left')
    merged_reviews_df['cluster_id'] = merged_reviews_df['cluster_id'].astype(object).fillna('UnknownSegment')
    
    original_segment_names_list = list(segments_df['cluster_id'].unique())


    # --- Aggregate Reviews and Generate Embeddings per Product (Multiprocessed) ---
    product_profiles = []
    unique_products = recent_reviews_df[['product_id', 'product_name', 'brand_name']].drop_duplicates().reset_index(drop=True)
    print(f"Preparing to process {len(unique_products)} unique products...")

    tasks_args_list = []
    for _, product_info_row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Preparing product tasks"):
        p_id = product_info_row['product_id']
        product_specific_reviews = merged_reviews_df[merged_reviews_df['product_id'] == p_id][
            ['user_rating', 'processed_text', 'cluster_id']
        ].copy()

        tasks_args_list.append((
            product_info_row, 
            p_id,
            product_specific_reviews,
            original_segment_names_list, 
            SENTENCE_MODEL_NAME,
            torch_device_str,
            MAX_AGGREGATE_TEXT_LEN_FOR_EMBEDDING,
            embedding_dim
        ))
    
    num_product_workers = os.cpu_count() if os.cpu_count() else 1
    print(f"Starting product profile generation with {num_product_workers} worker processes (HF_HUB_OFFLINE is '{os.environ.get('HF_HUB_OFFLINE')}')")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_product_workers) as executor:
        futures = [executor.submit(process_single_product, arg_set) for arg_set in tasks_args_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_args_list), desc="Generating product profiles"):
            try:
                profile = future.result()
                if profile:
                    product_profiles.append(profile)
            except Exception as e:
                print(f"An error occurred while processing a product task (see details below). Product might be skipped.")
                traceback.print_exc()


    if not product_profiles:
        print("No product profiles were generated. Check data, filters, or worker errors.")
         
    product_profiles_df = pd.DataFrame(product_profiles)
    if product_profiles_df.empty and product_profiles:
         print("Warning: DataFrame is empty but profiles list was not. Check profile structure from workers.")


    # --- Save Processed Data ---
    if not product_profiles_df.empty:
        print(f"Saving {len(product_profiles_df)} product profiles to {OUTPUT_PROFILES_PATH}...")
        try:
            output_dir = os.path.dirname(OUTPUT_PROFILES_PATH)
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
            product_profiles_df.to_parquet(OUTPUT_PROFILES_PATH, index=False)
            print(f"Product profiles saved to {OUTPUT_PROFILES_PATH}")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
            csv_fallback_path = OUTPUT_PROFILES_PATH.replace(".parquet", ".csv")
            print(f"Attempting to save as CSV instead to {csv_fallback_path} (embeddings will be stringified).")
            try:
                product_profiles_df.to_csv(csv_fallback_path, index=False)
                print(f"Saved as CSV to {csv_fallback_path}")
            except Exception as e_csv:
                print(f"Error saving CSV file: {e_csv}")
                traceback.print_exc()

    elif not product_profiles :
        print("No product profiles generated, nothing to save.")
    else:
        print("Product profiles list was populated, but the DataFrame is empty. Cannot save. Check profile structure from workers.")


    print(f"--- Preprocessing finished in {time.time() - start_script_time:.2f} seconds. ---")

if __name__ == '__main__':
    main()
