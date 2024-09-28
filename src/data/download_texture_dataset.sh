#!/bin/bash
DATA_DIR="data/raw"
mkdir $DATA_DIR

dataPoints=("cat_rescaled_rotated"
            "cat_dataset_v2_tiny" 
            "human" 
            "human_dataset_v2_tiny" 
            )

for dataPoint in ${dataPoints[@]}; do
    wget "https://vision.in.tum.de/webshare/g/intrinsic-neural-fields/data/${dataPoint}.zip" -P $DATA_DIR
    unzip "${DATA_DIR}/${dataPoint}.zip" -d $DATA_DIR
    rm -rf "${DATA_DIR}/${dataPoint}.zip"
done