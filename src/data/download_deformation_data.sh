#!/bin/bash

DATA_DIR="data/raw"
mkdir $DATA_DIR

dataPoints=("elephant-gallop"
            "face-poses" 
            )

for dataPoint in ${dataPoints[@]}; do
    wget "https://people.csail.mit.edu/sumner/research/deftransfer/data/${dataPoint}.zip" -P $DATA_DIR
    unzip "${DATA_DIR}/${dataPoint}.zip" -d $DATA_DIR
    rm -rf "${DATA_DIR}/${dataPoint}.zip"
done