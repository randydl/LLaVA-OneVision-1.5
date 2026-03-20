#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd $SCRIPT_DIR

export CONDA_DIR="$WORK_DIR/miniforge3"
. $CONDA_DIR/etc/profile.d/conda.sh
conda activate swift

cd ../examples_offline_packing/bmr_packing
python ./s1_bmr_sft_data_proc_indcoding.py
python ./s1_get_tokenlens_v4-sft.py
python ./2_do_hashbacket.py
python ./s2_prepare_rawsamples-bmr_sft_780k-8k-fast.py
python ./4_convert_packedsample_to_wds.py
