#!/bin/bash

DATA_DIR="data"
mkdir $DATA_DIR

## Diligent MV dataset ###
fileIdDiligent=18dheWmAxCNaBpYoH3usuFeH9vGlhODvx
fileNameDiligent=${DATA_DIR}/dataDiligent.zip

# Download diligent data with gdown
gdown --id $fileIdDiligent --output $fileNameDiligent
unzip "${fileNameDiligent}" -d $DATA_DIR

# Remove the zip file
rm ${fileNameDiligent}
#####################

### Simplified meshes ###
folderMeshes=$DATA_DIR/DiLiGenT-MV/simplifiedMeshes
mkdir $folderMeshes
fileIdMeshes=1cZvDGcO3z7-ySL8UZYjy-hQrC6EPRbsK
fileNameMeshes=${DATA_DIR}/dataSimplifiedMeshes.zip

# Download data with gdown
gdown --id $fileIdMeshes --output $fileNameMeshes
unzip "${fileNameMeshes}" -d $folderMeshes

# Remove the zip file
rm ${fileNameMeshes}
