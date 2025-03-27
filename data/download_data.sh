#!/bin/bash

DATASET_NAME="nadyinky/sephora-products-and-skincare-reviews"
RAW_DATA_PATH="data/raw"
ZIP_FILE="sephora-products-and-skincare-reviews.zip" #the zip file name

# Create the raw data directory if it doesn't exist
mkdir -p "$RAW_DATA_PATH"

# Download the dataset using Kaggle CLI
kaggle datasets download -d "$DATASET_NAME" -p "$RAW_DATA_PATH"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Kaggle dataset '$DATASET_NAME' downloaded successfully to '$RAW_DATA_PATH'"

  # Unzip the downloaded file
  unzip "$RAW_DATA_PATH/$ZIP_FILE" -d "$RAW_DATA_PATH"

  # Check if unzip was successful
  if [ $? -eq 0 ]; then
    echo "Dataset unzipped successfully."
    # Remove the zip file
    rm "$RAW_DATA_PATH/$ZIP_FILE"
  else
    echo "Error unzipping dataset."
    exit 1
  fi

else
  echo "Error downloading Kaggle dataset."
  exit 1
fi