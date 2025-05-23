from skincare_recommender.recommender import SkincareRecommenderV2 
import pandas as pd
import argparse
import time
import os

# --- Configuration for file paths (can be overridden by CLI args) ---
DEFAULT_PROFILES_PATH = 'data/processed/product_profile_embeddings_v4.parquet'
DEFAULT_SEGMENTS_PATH = 'notebooks/customer_segments_refined.csv'

def main(prompt, client_id, profiles_path, segments_path, top_n=5, num_threads=None): # Added num_threads
    """
    Main function to load preprocessed data, initialize RecommenderV2, and get recommendations.
    """
    print("--- Starting Skincare Recommendation Process (V2) ---")
    overall_start_time = time.time()

    if not os.path.exists(profiles_path):
        print(f"ERROR: Preprocessed product profiles not found at '{profiles_path}'.")
        print("Please run the 'preprocess_product_profiles.py' script first.")
        return
    
    if not os.path.exists(segments_path):
        print(f"ERROR: Client segments file not found at '{segments_path}'.")
        return

    # --- Initialize Recommender ---
    try:
        recommender = SkincareRecommenderV2(
            product_profiles_path=profiles_path,
            segments_data_path=segments_path,
            num_threads=num_threads # Pass num_threads to the recommender
        )
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Get Recommendations ---
    print(f"\nRequesting recommendations for Client ID {client_id} with prompt: '{prompt}'")
    recommendations = recommender.get_recommendations(
        prompt=prompt,
        client_id=str(client_id), 
        top_n=top_n
    )

    # --- Display Recommendations ---
    if recommendations:
        print(f"\n--- Top {len(recommendations)} Recommendations (V2) ---")
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. Product: {rec.get('product_name', 'N/A')} ({rec.get('brand_name', 'N/A')})")
            print(f"   Score: {rec.get('score', 0.0):.4f} (Recent Reviews: {rec.get('total_recent_reviews',0)})") # Added total_recent_reviews display
            print(f"   User's Segment ({recommender.get_client_segment(client_id)}) Avg Rating: {rec.get('segment_avg_rating_for_user', 'N/A')}"
                  f" (from {rec.get('segment_review_count_for_user', 0)} reviews in segment)")
            
            pros = rec.get('pros_summary_personalized', 'No specific pros highlighted.')
            pros_display = (pros[:250] + '...') if len(pros) > 250 else pros
            print(f"   PROS: {pros_display}")

            cons = rec.get('cons_summary_personalized', 'No specific cons highlighted.')
            cons_display = (cons[:250] + '...') if len(cons) > 250 else cons
            print(f"   CONS: {cons_display}")
    else:
        print("No recommendations found for the given prompt and client.")
    
    print(f"\n--- Skincare Recommendation Process (V2) Finished in {time.time() - overall_start_time:.2f}s ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Skincare Product Recommender V2 (using preprocessed profiles)")
    parser.add_argument("--prompt", type=str, required=True, help="Your skincare concern or what you are looking for.")
    parser.add_argument("--client_id", type=str, required=True, help="The client ID for personalized recommendations.")
    parser.add_argument("--profiles_path", type=str, default=DEFAULT_PROFILES_PATH, help="Path to the preprocessed product profiles Parquet file.")
    parser.add_argument("--segments_path", type=str, default=DEFAULT_SEGMENTS_PATH, help="Path to the client segments CSV file.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top recommendations to return.")
    parser.add_argument("--num_threads", type=int, default=None, help="Number of threads for scoring (default: auto).")


    args = parser.parse_args()

    main(
        prompt=args.prompt,
        client_id=args.client_id,
        profiles_path=args.profiles_path,
        segments_path=args.segments_path,
        top_n=args.top_n,
        num_threads=args.num_threads
    )
