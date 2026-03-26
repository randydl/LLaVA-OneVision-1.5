#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TP="${TP:-1}"
PP="${PP:-1}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b/auto-model}"
MCORE_CHECKPOINT_PATH="${MCORE_CHECKPOINT_PATH:-}"
PREPROCESSOR_PATH="${PREPROCESSOR_PATH:-$HF_MODEL_PATH}"
TEST_IMAGE_PATH="${TEST_IMAGE_PATH:-${REPO_ROOT}/asset/performance.png}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE=$((TP * PP))

export AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_ROOT}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${REPO_ROOT}/aiak_megatron}"

if [[ ! -d "$HF_MODEL_PATH" ]]; then
    echo "HF model path does not exist: $HF_MODEL_PATH"
    exit 1
fi

if [[ ! -f "$TEST_IMAGE_PATH" ]]; then
    echo "Test image path does not exist: $TEST_IMAGE_PATH"
    exit 1
fi

cd "$REPO_ROOT"

if [[ -z "$MCORE_CHECKPOINT_PATH" ]]; then
    MCORE_CHECKPOINT_PATH="$REPO_ROOT/tmp_test_mcore_ckpt_tp${TP}_pp${PP}"
    echo "Converting HF->mcore checkpoint to $MCORE_CHECKPOINT_PATH"
    bash examples/llava_onevision2/convert/convert_4b_hf_to_mcore.sh \
        "$HF_MODEL_PATH" \
        "$MCORE_CHECKPOINT_PATH" \
        "$TP" \
        "$PP"
fi

if [[ ! -d "$MCORE_CHECKPOINT_PATH" ]]; then
    echo "Mcore checkpoint path does not exist: $MCORE_CHECKPOINT_PATH"
    exit 1
fi

export HF_MODEL_PATH
export MCORE_CHECKPOINT_PATH
export PREPROCESSOR_PATH
export TEST_IMAGE_PATH
export CONSISTENCY_TEST_TP="$TP"
export CONSISTENCY_TEST_PP="$PP"

PYTHONPATH="ds/llavaonevision2:$AIAK_MAGATRON_PATH:$REPO_ROOT:${PYTHONPATH:-}" \
torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    -m pytest tests/test_model_consistency.py \
    -v \
    "$@"
