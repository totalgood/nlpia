#!/usr/bin/env bash
# download.sh

BIGDATA_PATH=$(python -c 'from nlpia.data import BIGDATA_PATH; print(BIGDATA_PATH);')

oldIFS="$IFS"
IFS=$'\n' BIG_URLS=($(python -c 'from nlpia.data import BIG_URLS; print("\n".join([u[0] for u in BIG_URLS.values()]));'))
IFS="$oldIFS"

for URL in "${BIG_URLS[@]}"; do
    TARGET_FILE="$URL"
    if [ "${URL: -5: 5}" = "?dl=0" ]; then
        TARGET_FILE="${URL: : -5}"
    fi
    TARGET_FILE=$(basename "$TARGET_FILE")
    if [ -f "$TARGET_FILE" ]; then
        echo "File exists. Delete $BIGDATA_PATH/$TARGET_FILE if you'd like to overwrite it."
    else
        echo "Downloading: $URL to $BIGDATA_PATH/$TARGET_FILE"
        wget -O "$BIGDATA_PATH/$TARGET_FILE" "$URL"
        mv $(basename "$URL") "$TARGET_FILE"
    fi
done

