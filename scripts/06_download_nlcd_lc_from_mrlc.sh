#!/usr/bin/env bash

# Set your output directory
OUTDIR="/data/NLCD"
mkdir -p "$OUTDIR"

# Base URL pattern
# Download the NLCD from MRLC for each year and unzip.
BASE_URL="https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles"

# Loop through the years
for YEAR in {2010..2024}; do
    FILE="Annual_NLCD_LndCov_${YEAR}_CU_C1V1.zip"
    URL="${BASE_URL}/${FILE}"
    DEST="${OUTDIR}/${FILE}"

    echo "Downloading ${FILE}..."
    curl -L -o "$DEST" "$URL"

    if [ $? -eq 0 ]; then
        echo "Unzipping ${FILE}..."
        unzip -o "$DEST" -d "$OUTDIR/${YEAR}/"
    else
        echo "Failed to download ${FILE}"
    fi
done

echo "All downloads (2010–2024) completed."
