#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd $SCRIPT_DIR

export CONDA_DIR="$WORK_DIR/miniforge3"
. $CONDA_DIR/etc/profile.d/conda.sh
conda activate swift

python tools/data_preprocess/convert_to_webdataset.py \
    --jsonl_dir=/nas_train/app.e0016372/datasets/LLaVA-OneVision-1.5-Instruct-Data/files \
    --image_dir=/nas_train/app.e0016372/datasets/LLaVA-OneVision-1.5-Instruct-Data/images \
    --output_dir=/nas_train/app.e0016372/datasets/LLaVA-OneVision-1.5-Instruct-Data/webdataset
