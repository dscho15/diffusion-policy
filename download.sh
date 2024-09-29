#!/bin/bash

# Define the folder where you want to save the images
download_folder="imgs"  # Replace this with the path to your folder

# Create the folder if it doesn't exist
mkdir -p "$download_folder"

# Define the path to the .csv file with the URLs
csv_file="urls.csv"  # Replace with the path to your CSV file

# Loop through each URL in the CSV file and download it
while IFS=, read -r url; do
    wget -P "$download_folder" "$url"
done < "$csv_file"

echo "All files have been downloaded to $download_folder"