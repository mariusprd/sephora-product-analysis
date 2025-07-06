# Sephora Product Analysis

![Dataset Cover](assets/dataset-cover.jpg)

This repository contains the code and resources for a personalized product recommendation system for Sephora, focusing on skincare. [cite_start]The system analyzes product descriptions and customer reviews to segment clients, generate tailored recommendations, and provide insightful PROS/CONS summaries for each suggested product. [cite: 1, 3, 129, 131]

## Project Objectives

The primary objective was to create an intelligent recommendation system that suggests products most relevant to a user's specific skincare concerns (e.g., "oily skin," "dry skin") and their client profile. [cite_start]It aims to provide clear explanations for recommendations through automatically generated "PROS" and "CONS" summaries from aggregated review texts. [cite: 127, 129, 131, 132]

## Features & Functionality

* [cite_start]**Personalized Recommendations:** Suggests top 5 products based on user input (skincare concern) and client segment. [cite: 129, 152, 153, 260]
* [cite_start]**Client Segmentation:** Divides customers into distinct segments using K-Means clustering, based on purchasing behavior and product interactions. [cite: 135, 158, 241]
* [cite_start]**Review Analysis & Summarization:** Processes vast amounts of customer reviews to extract semantic insights and generate concise "PROS" and "CONS" summaries for recommended products. [cite: 131, 132, 144, 155, 268]
* [cite_start]**Data Preprocessing Pipeline:** Includes cleaning, tokenizing, and lemmatizing review text (using NLTK) [cite: 74, 75, 139][cite_start], as well as aggregating reviews and calculating segment-specific statistics. [cite: 79, 80, 81, 82, 83]

## Recommendation Flow & Architecture

The system operates through a multi-stage pipeline, from raw data processing to personalized recommendation generation.

![Recommendation Flow Diagram](assets/flow.png)

*(Note: Please ensure `flow.png` is placed in the `assets/` directory.)*

## Implementation Details

### Dataset
[cite_start]We utilized a Kaggle dataset comprising Sephora product information and customer reviews, which were aggregated using product IDs for comprehensive analysis. [cite: 15, 16, 40]

### Data Preprocessing
Raw review texts underwent cleaning (lowercase conversion, punctuation removal) and lemmatization using NLTK for standardization. Positive and negative reviews for each product were aggregated and capped to a manageable length. [cite_start]Additionally, segment-specific statistics like average user ratings and review counts were calculated for each product. [cite: 72, 73, 74, 75, 79, 80, 81, 82, 83]

### Client Segmentation
K-Means clustering was employed for client segmentation, with the optimal number of clusters (`k`) determined using silhouette analysis to ensure distinct and meaningful segments. [cite_start]This analysis identified 14 customer segments, including 'Contented Loyalists' (82,734 customers) and 'High-Spending, Prolific Reviewing Brand Connoisseurs' (4,263 customers). [cite: 135, 158, 160, 162, 166, 241]

### Semantic Analysis & Summarization
Review texts and user inputs were converted into numerical embeddings using the `all-MiniLM-L6-v2` model for semantic understanding. [cite_start]For generating "PROS" and "CONS" summaries, the `distilbart-cnn-12-6` model was used to provide concise explanations for recommended products. [cite: 145, 150, 155]

### Recommendation Logic
[cite_start]Product scores are refined based on how well a product matches a user's concern, its rating within the user's client segment, and the total number of reviews (as an indicator of trustworthiness). [cite: 264, 265, 267]

## Final Results & Limitations

The project yielded very promising results in generating personalized recommendations. [cite_start]Identified limitations include computational power constraints and the quality of the reasoning summaries for certain recommendations. [cite: 272, 273]

## Technology Stack

* [cite_start]**Data Analysis & ML:** Python, NLTK, scikit-learn (K-Means), `all-MiniLM-L6-v2` (for embeddings), `distilbart-cnn-12-6` (for summarization) [cite: 75, 135, 145, 155, 158]
* [cite_start]**Data Source:** Kaggle [cite: 15]

## Setup

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/SephoraProductAnalysis.git](https://github.com/your-username/SephoraProductAnalysis.git)
    cd SephoraProductAnalysis
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Download the Sephora dataset:

    * Make sure you have the Kaggle CLI installed and configured. If not, follow the instructions on the [Kaggle CLI GitHub page](https://github.com/Kaggle/kaggle-api).
    * IMPORTANT: You need to have the kaggle.json file in the `~/.kaggle` directory.
    * Run the download script:

        ```bash
        ./data/download_data.sh
        ```
