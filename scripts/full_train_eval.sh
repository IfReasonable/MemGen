#!/bin/bash
set -euo pipefail

# ===== Environment =====
export DEBUG_MODE=true
export LOG_PATH="./debug_full_pipeline.txt"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"  # adjust GPUs
# Use different ports per stage to avoid rare re-launch hangs.
export MAIN_PROCESS_PORT=29507
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

# ===== Models =====
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SHORT="${REASONER_MODEL#*/}"

# ===== Dataset =====
DATASET_NAME="gpqa"  # gsm8k, gpqa, kodcode, triviaqa

# ===== MemGen configs =====
WEAVER_TRAIN_METHOD="sft"   # sft or grpo
TRIGGER_TRAIN_METHOD="grpo" # grpo only

# ===== Augmentation configs =====
# For gsm8k/gpqa/kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5 (paper default)
# For triviaqa:          MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

# ===== Train/Eval settings =====
BATCH_SIZE=1
EVAL_BATCH_SIZE=4
TEMPERATURE=0.0

# ===== GRPO settings =====
# TRL GRPO requires: generation_batch_size divisible by num_generations.
# generation_batch_size is typically: WORLD_SIZE * per_device_train_batch_size * gradient_accumulation_steps.
TRIGGER_GRAD_ACCUM_STEPS=4
TRIGGER_DESIRED_NUM_GENERATIONS=8

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# Helper: pick the largest factor of $1 that is <= $2 (fallback to 1)
pick_num_generations() {
    local gen_batch_size="$1"
    local desired="$2"
    local best=1
    local i
    for ((i=1; i<=desired; i++)); do
        if (( gen_batch_size % i == 0 )); then
            best=$i
        fi
    done
    echo "$best"
}

WEAVER_PORT=${MAIN_PROCESS_PORT}
TRIGGER_PORT=$((MAIN_PROCESS_PORT + 1))
EVAL_PORT=$((MAIN_PROCESS_PORT + 2))

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Helper: find latest run dir matching the current aug config
find_latest_run_dir() {
    local mode="$1"  # train or evaluate
    local parent="${ROOT_DIR}/results/${mode}/${DATASET_NAME}/${MODEL_SHORT}"
    local pattern="${parent}/pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_*"
    ls -td ${pattern} 2>/dev/null | head -n 1
}

# ===== 1) Train Weaver =====
# echo "[$(ts)] Stage 1/3: train weaver (${WEAVER_TRAIN_METHOD})"
# python -m accelerate.commands.launch \
#     --config_file=configs/zero2.yaml \
#     --num_processes ${NUM_PROCESSES} \
#     --main_process_port ${WEAVER_PORT} \
#     main.py \
#     --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
#     --options \
#     model.model_name ${REASONER_MODEL} \
#     model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
#     model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
#     model.weaver.model_name ${WEAVER_MODEL} \
#     model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
#     model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
#     model.trigger.model_name ${TRIGGER_MODEL} \
#     model.trigger.active False \
#     datasets.mode ${WEAVER_TRAIN_METHOD} \
#     run.mode train \
#     run.train_weaver True \
#     run.train_trigger False \
#     run.train_weaver_method ${WEAVER_TRAIN_METHOD} \
#     run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
#     run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
#     run.weaver.sft.bf16 True \
#     run.interaction.do_sample True \
#     run.interaction.temperature 1.0 \
#     run.interaction.max_response_length 1024

# WEAVER_RUN_DIR="$(find_latest_run_dir train)"
# WEAVER_CKPT="${WEAVER_RUN_DIR}/weaver/model.safetensors"
# if [[ ! -f "${WEAVER_CKPT}" ]]; then
#     echo "Weaver checkpoint not found: ${WEAVER_CKPT}" >&2
#     exit 1
# fi
# echo "[$(ts)] Weaver done. ckpt=${WEAVER_CKPT}"

# # ===== 2) Train Trigger (load Weaver) =====
# echo "[$(ts)] Stage 2/3: train trigger (${TRIGGER_TRAIN_METHOD})"

# # Ensure TRL GRPO constraint holds.
# TRIGGER_GEN_BATCH_SIZE=$((NUM_PROCESSES * BATCH_SIZE * TRIGGER_GRAD_ACCUM_STEPS))
# TRIGGER_NUM_GENERATIONS="${TRIGGER_DESIRED_NUM_GENERATIONS}"
# if (( TRIGGER_GEN_BATCH_SIZE % TRIGGER_NUM_GENERATIONS != 0 )); then
#     TRIGGER_NUM_GENERATIONS="$(pick_num_generations "${TRIGGER_GEN_BATCH_SIZE}" "${TRIGGER_DESIRED_NUM_GENERATIONS}")"
#     echo "[$(ts)] [WARN] TRL GRPO requires generation_batch_size divisible by num_generations." >&2
#     echo "[$(ts)] [WARN] generation_batch_size=${TRIGGER_GEN_BATCH_SIZE}, desired_num_generations=${TRIGGER_DESIRED_NUM_GENERATIONS} -> using num_generations=${TRIGGER_NUM_GENERATIONS}" >&2
# fi

# python -m accelerate.commands.launch \
#     --config_file=configs/zero2.yaml \
#     --num_processes ${NUM_PROCESSES} \
#     --main_process_port ${TRIGGER_PORT} \
#     main.py \
#     --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
#     --options \
#     model.model_name ${REASONER_MODEL} \
#     model.load_model_path ${WEAVER_CKPT} \
#     model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
#     model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
#     model.weaver.model_name ${WEAVER_MODEL} \
#     model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
#     model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
#     model.trigger.model_name ${TRIGGER_MODEL} \
#     model.trigger.active True \
#     datasets.mode ${TRIGGER_TRAIN_METHOD} \
#     run.mode train \
#     run.train_weaver False \
#     run.train_trigger True \
#     run.train_trigger_method ${TRIGGER_TRAIN_METHOD} \
#     run.trigger.grpo.per_device_train_batch_size ${BATCH_SIZE} \
#     run.trigger.grpo.per_device_eval_batch_size ${BATCH_SIZE} \
#     run.trigger.grpo.num_train_epochs 1 \
#     run.trigger.grpo.num_generations ${TRIGGER_NUM_GENERATIONS} \
#     run.trigger.grpo.gradient_accumulation_steps ${TRIGGER_GRAD_ACCUM_STEPS} \
#     run.interaction.do_sample True \
#     run.interaction.temperature 1.0 \
#     run.interaction.max_response_length 1024

TRIGGER_RUN_DIR="$(find_latest_run_dir train)"
TRIGGER_CKPT="${TRIGGER_RUN_DIR}/trigger/model.safetensors"
if [[ ! -f "${TRIGGER_CKPT}" ]]; then
    echo "Trigger checkpoint not found: ${TRIGGER_CKPT}" >&2
    exit 1
fi
echo "[$(ts)] Trigger done. ckpt=${TRIGGER_CKPT}"

# ===== 3) Evaluate MemGen (Weaver + Trigger) =====
echo "[$(ts)] Stage 3/3: evaluate full MemGen (weaver+trigger)"
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes ${NUM_PROCESSES} \
    --main_process_port ${EVAL_PORT} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${TRIGGER_CKPT} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active True \
    run.mode evaluate \
    run.interaction.batch_size ${EVAL_BATCH_SIZE} \
    run.interaction.do_sample False \
    run.interaction.temperature ${TEMPERATURE} \
    run.interaction.max_response_length 1024

echo "Done. Weaver ckpt: ${WEAVER_CKPT}"
echo "Done. Trigger ckpt: ${TRIGGER_CKPT}"
