import os
# It's crucial to set HF_HUB_OFFLINE for child processes.
# We will set it to '0' (or unset it) for the initial download,
# then set it to '1' before starting the multiprocessing pool.

import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# SentenceTransformer will be imported later, after managing HF_HUB_OFFLINE
import torch
import concurrent.futures
from tqdm import tqdm
from pandarallel import pandarallel
import traceback # For detailed error printing

# --- Configuration ---
REVIEWS_FILE_PATH = 'data/processed/reviews.csv'
SEGMENTS_FILE_PATH = 'notebooks/customer_segments_refined.csv' # Or your final segments file
OUTPUT_PROFILES_PATH = 'data/processed/product_profile_embeddings.parquet'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_AGGREGATE_TEXT_LEN = 2000

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
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    filtered_tokens = [token for token in lemmatized_tokens if token.isalnum()]
    return " ".join(filtered_tokens)

def process_single_product(args):
    from sentence_transformers import SentenceTransformer # Import here for the worker process

    product_info_series, product_id, product_specific_reviews_df, \
    original_segment_names, model_name, device_str, max_text_len, \
    embedding_dim_val = args
    
    # This initialization should now use the cache and not hit the network
    # because HF_HUB_OFFLINE=1 should be set in the environment.
    sbert_model_worker = SentenceTransformer(model_name, device=device_str)

    if product_specific_reviews_df.empty:
        return None

    product_specific_reviews_df['user_rating'] = pd.to_numeric(product_specific_reviews_df['user_rating'], errors='coerce')
    valid_ratings_df = product_specific_reviews_df.dropna(subset=['user_rating'])

    good_reviews_text = " ".join(valid_ratings_df[valid_ratings_df['user_rating'] >= 4]['processed_text'].tolist())
    bad_reviews_text = " ".join(valid_ratings_df[valid_ratings_df['user_rating'] <= 2]['processed_text'].tolist())

    positive_aggregate_text = good_reviews_text[:max_text_len] if good_reviews_text else "no positive reviews"
    negative_aggregate_text = bad_reviews_text[:max_text_len] if bad_reviews_text else "no negative reviews"

    pro_embedding = sbert_model_worker.encode(positive_aggregate_text).tolist() \
        if positive_aggregate_text != "no positive reviews" else np.zeros(embedding_dim_val).tolist()
    con_embedding = sbert_model_worker.encode(negative_aggregate_text).tolist() \
        if negative_aggregate_text != "no negative reviews" else np.zeros(embedding_dim_val).tolist()

    segment_stats = {}
    for segment_name in original_segment_names:
        segment_reviews = product_specific_reviews_df[product_specific_reviews_df['cluster_id'] == segment_name]
        segment_reviews_numeric_rating = segment_reviews.copy()
        segment_reviews_numeric_rating['user_rating'] = pd.to_numeric(segment_reviews_numeric_rating['user_rating'], errors='coerce')
        segment_reviews_numeric_rating.dropna(subset=['user_rating'], inplace=True)

        if not segment_reviews_numeric_rating.empty:
            segment_stats[f'segment_{segment_name}_avg_rating'] = segment_reviews_numeric_rating['user_rating'].mean()
            segment_stats[f'segment_{segment_name}_review_count'] = len(segment_reviews) 
        else:
            segment_stats[f'segment_{segment_name}_avg_rating'] = 0.0
            segment_stats[f'segment_{segment_name}_review_count'] = 0

    if 'UnknownSegment' not in original_segment_names and 'UnknownSegment' in product_specific_reviews_df['cluster_id'].unique():
        segment_reviews_unknown = product_specific_reviews_df[product_specific_reviews_df['cluster_id'] == 'UnknownSegment']
        segment_reviews_unknown_numeric_rating = segment_reviews_unknown.copy()
        segment_reviews_unknown_numeric_rating['user_rating'] = pd.to_numeric(segment_reviews_unknown_numeric_rating['user_rating'], errors='coerce')
        segment_reviews_unknown_numeric_rating.dropna(subset=['user_rating'], inplace=True)

        if not segment_reviews_unknown_numeric_rating.empty:
            segment_stats[f'segment_UnknownSegment_avg_rating'] = segment_reviews_unknown_numeric_rating['user_rating'].mean()
            segment_stats[f'segment_UnknownSegment_review_count'] = len(segment_reviews_unknown)

    profile = {
        'product_id': product_id,
        'product_name': product_info_series['product_name'],
        'brand_name': product_info_series['brand_name'],
        'positive_aggregate_text': positive_aggregate_text,
        'negative_aggregate_text': negative_aggregate_text,
        'pro_embedding': pro_embedding,
        'con_embedding': con_embedding,
        **segment_stats
    }
    return profile

def main():
    print("--- Starting Offline Product Profile Preprocessing (Optimized v3) ---")
    start_script_time = time.time()
    
    # Import SentenceTransformer here
    from sentence_transformers import SentenceTransformer

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device} for sentence embeddings.")

    # --- Step 1: Ensure Model is Cached (Online Mode) ---
    print(f"Ensuring sentence transformer model '{SENTENCE_MODEL_NAME}' is cached locally...")
    # Temporarily ensure HF_HUB_OFFLINE is NOT '1' for this download/cache step
    original_hf_offline_status = os.environ.pop('HF_HUB_OFFLINE', None)

    embedding_dim = 0 # Initialize
    try:
        sbert_temp_for_caching = SentenceTransformer(SENTENCE_MODEL_NAME, device=device)
        embedding_dim = sbert_temp_for_caching.get_sentence_embedding_dimension()
        del sbert_temp_for_caching # Free memory
        print(f"Model '{SENTENCE_MODEL_NAME}' (dim: {embedding_dim}) is available/cached.")
    except Exception as e:
        print(f"FATAL: Could not download/cache model '{SENTENCE_MODEL_NAME}'. Exiting. Error: {e}")
        traceback.print_exc()
        return
    finally:
        # Restore original HF_HUB_OFFLINE status if it was set, or leave it unset
        if original_hf_offline_status is not None:
            os.environ['HF_HUB_OFFLINE'] = original_hf_offline_status
        else: # If it wasn't set before, ensure it's not '1' unless we set it later
            if 'HF_HUB_OFFLINE' in os.environ:
                 del os.environ['HF_HUB_OFFLINE']


    # --- Step 2: Set to OFFLINE mode for all subsequent operations ---
    print("Setting Hugging Face Hub to OFFLINE mode for worker processes.")
    os.environ['HF_HUB_OFFLINE'] = '1'

    # --- Load Data ---
    print(f"Loading reviews from {REVIEWS_FILE_PATH}...")
    try:
        dtype_spec = {'author_id': str, 'user_rating': str}
        print("Attempting to load CSV with Python engine...")
        reviews_df = pd.read_csv(REVIEWS_FILE_PATH, dtype=dtype_spec, engine='python')
        reviews_df['user_rating'] = pd.to_numeric(reviews_df['user_rating'], errors='coerce')
        reviews_df.dropna(subset=['user_rating'], inplace=True)
    except FileNotFoundError:
        print(f"ERROR: Reviews file not found at {REVIEWS_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during data loading: {e}")
        traceback.print_exc()
        return

    required_review_cols = ['author_id', 'product_id', 'product_name', 'brand_name', 'user_rating', 'review_text', 'submission_time']
    if not all(col in reviews_df.columns for col in required_review_cols):
        print(f"ERROR: Reviews CSV must contain columns: {', '.join(required_review_cols)}")
        missing_cols = [col for col in required_review_cols if col not in reviews_df.columns]
        print(f"Missing: {missing_cols}")
        return

    print(f"Loading client segments from {SEGMENTS_FILE_PATH}...")
    try:
        segments_df = pd.read_csv(SEGMENTS_FILE_PATH) 
        if 'author_id' not in segments_df.columns:
            if 'customer_id' in segments_df.columns:
                print("Renaming 'customer_id' to 'author_id' in segments data.")
                segments_df.rename(columns={'customer_id': 'author_id'}, inplace=True)
            else:
                print("ERROR: 'author_id' (or 'customer_id') column is missing in segments data.")
                return
        if 'cluster_id' not in segments_df.columns:
            print("ERROR: 'cluster_id' column is missing in segments data.")
            return
        segments_df['author_id'] = segments_df['author_id'].astype(str)
        segments_df['cluster_id'] = segments_df['cluster_id'].astype(str)
    except FileNotFoundError:
        print(f"Warning: Segments file not found at {SEGMENTS_FILE_PATH}. Proceeding without explicit segmentation.")
        segments_df = pd.DataFrame(columns=['author_id', 'cluster_id']) 
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during segment data loading: {e}")
        traceback.print_exc()
        return

    print("Filtering reviews from the last 2 years...")
    try:
        reviews_df['submission_time'] = pd.to_datetime(reviews_df['submission_time'], errors='coerce')
        reviews_df.dropna(subset=['submission_time'], inplace=True)
    except Exception as e:
        print(f"Error converting 'submission_time' to datetime: {e}. Please check the date format.")
        return

    two_years_ago = datetime.now() - timedelta(days=4*365)
    recent_reviews_df = reviews_df[reviews_df['submission_time'] >= two_years_ago].copy()
    print(f"Found {len(recent_reviews_df)} reviews from the last 2 years (out of {len(reviews_df)} total).")
    if recent_reviews_df.empty:
        print("No recent reviews found. Exiting.")
        return

    print("Preprocessing review text for aggregation (using pandarallel)...")
    num_text_proc_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Initializing pandarallel with {num_text_proc_workers} workers.")
    # pandarallel might itself spawn processes that need to know about HF_HUB_OFFLINE.
    # It should inherit the env var set before its initialization.
    pandarallel.initialize(nb_workers=num_text_proc_workers, progress_bar=True, verbose=0)
    recent_reviews_df['processed_text'] = recent_reviews_df['review_text'].parallel_apply(preprocess_text_for_aggregation)

    print("Merging reviews with segment data...")
    recent_reviews_df['author_id'] = recent_reviews_df['author_id'].astype(str)
    if not segments_df.empty:
        merged_reviews_df = pd.merge(recent_reviews_df, segments_df, on='author_id', how='left')
        merged_reviews_df['cluster_id'] = merged_reviews_df['cluster_id'].astype(object).fillna('UnknownSegment')
    else:
        merged_reviews_df = recent_reviews_df.copy()
        merged_reviews_df['cluster_id'] = 'UnknownSegment'
    
    original_segment_names_list = list(segments_df['cluster_id'].unique()) if not segments_df.empty else []

    print(f"Using pre-obtained embedding dimension: {embedding_dim}")

    product_profiles = []
    unique_products = recent_reviews_df[['product_id', 'product_name', 'brand_name']].drop_duplicates()
    print(f"Preparing to process {len(unique_products)} unique products...")

    tasks_args_list = []
    for _, product_info_row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Preparing product tasks"):
        p_id = product_info_row['product_id']
        product_specific_reviews = merged_reviews_df[merged_reviews_df['product_id'] == p_id][
            ['user_rating', 'processed_text', 'cluster_id'] 
        ].copy()
        tasks_args_list.append((
            product_info_row, p_id, product_specific_reviews, original_segment_names_list, 
            SENTENCE_MODEL_NAME, device, MAX_AGGREGATE_TEXT_LEN, embedding_dim
        ))
    
    num_product_workers = os.cpu_count() if os.cpu_count() else 1
    print(f"Starting product profile generation with {num_product_workers} worker processes (HF_HUB_OFFLINE is now '1')...")

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
         print("Warning: DataFrame is empty but profiles list was not. Check profile structure.")

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
            print(f"Attempting to save as CSV instead to {csv_fallback_path}.")
            try:
                product_profiles_df.to_csv(csv_fallback_path, index=False)
                print(f"Saved as CSV to {csv_fallback_path}")
            except Exception as e_csv:
                print(f"Error saving CSV file: {e_csv}")
                traceback.print_exc()
    elif not product_profiles :
        print("No product profiles generated, nothing to save.")
    else: # product_profiles not empty, but dataframe is
        print("Product profiles list was populated, but the DataFrame is empty. Cannot save. Check profile structure from workers.")

    print(f"--- Preprocessing finished in {time.time() - start_script_time:.2f} seconds. ---")

if __name__ == '__main__':
    main()