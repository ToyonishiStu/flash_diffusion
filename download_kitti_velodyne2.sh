#!/bin/bash
# Download KITTI Raw Data - velodyne_points only (no images)
# Source: https://github.com/Deepak3994/Kitti-Dataset

BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
OUTPUT_DIR="${1:-./kitti_raw}"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit 1

# Calibration files
calib_files=(2011_09_26_calib.zip)

# All drive sequences
drives=(2011_09_26_drive_0001)

# Download calibration files
echo "=== Downloading calibration files ==="
for calib in "${calib_files[@]}"; do
    echo "Downloading: $calib"
    wget -c "${BASE_URL}/${calib}" -O "$calib"
    unzip -o "$calib"
    rm "$calib"
done

# Download velodyne_points only for each drive
echo "=== Downloading velodyne_points ==="
for drive in "${drives[@]}"; do
    filename="${drive}_sync.zip"
    url="${BASE_URL}/${drive}/${filename}"

    echo "Downloading: $filename"
    wget -c "$url" -O "$filename"
    if [ $? -eq 0 ]; then
        unzip -o "$filename" "*/velodyne_points/*"
        rm "$filename"
    else
        echo "ERROR: Failed to download $filename — skipping"
        rm -f "$filename"
    fi
done

echo "=== Done ==="
