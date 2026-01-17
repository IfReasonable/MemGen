#!/bin/bash
set -euo pipefail

export DEBUG_MODE=true
export LOG_PATH="./debug_eval_vanilla.txt"

# GPUs / distributed
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"   # adjust, e.g. "0,1,2,3"
export MAIN_PROCESS_PORT=29517
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Derive process count from CUDA_VISIBLE_DEVICES (e.g. "1,2,3" -> 3).
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_PROCESSES=1
fi

# ===== Model / Dataset =====
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME="gpqa"  # gsm8k, gpqa, kodcode, triviaqa

# ===== Vanilla settings =====
# No extra weights loaded; no prompt/inference augmentation; trigger disabled.
MAX_PROMPT_AUG_NUM=0
MAX_INFERENCE_AUG_NUM=0
TRIGGER_ACTIVE=False
EVAL_BATCH_SIZE=$((1*NUM_PROCESSES))
TEMPERATURE=0.0
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes ${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    run.mode evaluate \
    run.eval_vanilla True \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path null \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.interaction.batch_size ${EVAL_BATCH_SIZE} \
    run.interaction.do_sample False \
    run.interaction.temperature ${TEMPERATURE} \
    run.interaction.max_response_length 1024
