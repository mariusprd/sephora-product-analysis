# Sephora Product Analysis

![Dataset Cover](assets/dataset-cover.jpg)

This repository contains the code and resources for analyzing Sephora products based on descriptions and reviews.

## Project Description

Sephora products analysis using descriptions and reviews. With this the project tries to segment the clients, give promotion suggestions and also recommend products.

## Dataset

The dataset used in this project is from Kaggle: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews).

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
