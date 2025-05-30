#!/bin/bash

# Create the datasets directory if it doesn't exist
mkdir -p datasets
mkdir -p datasets/dsm
mkdir -p datasets/dtm


# Download the file into the datasets directory
# curl -sS -L "https://api.ellipsis-drive.com/v3/path/a4a8a27b-e36e-4dd5-a75b-f7b6c18d33ec/raster/timestamp/fc9d369f-94ca-4373-8281-a6854edb67c9/file/8b9df1d9-bc21-4a1b-a51b-e09840a637f7/data" -o datasets/dsm/2024_R_51BZ2.TIF &
# curl -sS -L "https://api.ellipsis-drive.com/v3/path/a4a8a27b-e36e-4dd5-a75b-f7b6c18d33ec/raster/timestamp/fc9d369f-94ca-4373-8281-a6854edb67c9/file/f86d47f8-2983-4494-9786-c4a597d3cd34/data" -o datasets/dsm/2024_R_51EZ1.TIF &
# curl -sS -L "https://api.ellipsis-drive.com/v3/path/a4a8a27b-e36e-4dd5-a75b-f7b6c18d33ec/raster/timestamp/fc9d369f-94ca-4373-8281-a6854edb67c9/file/de55eb49-6547-430b-b196-f4abf5009598/data" -o datasets/dsm/2024_R_51DN2.TIF &
# curl -sS -L "https://api.ellipsis-drive.com/v3/path/a4a8a27b-e36e-4dd5-a75b-f7b6c18d33ec/raster/timestamp/fc9d369f-94ca-4373-8281-a6854edb67c9/file/7fc75f0d-33a7-495c-bbfa-f992e96925c8/data" -o datasets/dsm/2024_R_51GN1.TIF &

curl -sS -L "https://fsn1.your-objectstorage.com/hwh-ahn/AHN5/02b_DTM_5m/2024_M5_51GN1.TIF" -o datasets/dtm/2024_M5_51GN1.TIF &
curl -sS -L "https://fsn1.your-objectstorage.com/hwh-ahn/AHN5/02b_DTM_5m/2024_M5_51EZ1.TIF" -o datasets/dtm/2024_M5_51EZ1.TIF &
curl -sS -L "https://fsn1.your-objectstorage.com/hwh-ahn/AHN5/02b_DTM_5m/2024_M5_51BZ2.TIF" -o datasets/dtm/2024_M5_51BZ2.TIF &
curl -sS -L "https://fsn1.your-objectstorage.com/hwh-ahn/AHN5/02b_DTM_5m/2024_M5_51DN2.TIF" -o datasets/dtm/2024_M5_51DN2.TIF &

echo "Downloading..."

wait

echo "All downloads completed."