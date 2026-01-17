#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6"
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Derive process count from CUDA_VISIBLE_DEVICES (e.g. "1,2,3" -> 3).
# This avoids being limited by configs/zero2.yaml's num_processes.
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_PROCESSES=1
fi

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 

# Dataset configs
# DATASET_NAME="gsm8k"  # options: gsm8k, gpqa, kodcode, triviaqa
DATASET_NAME="gpqa"  # options: gsm8k, gpqa, kodcode, triviaqa
DATASET_MODE="grpo"   # options: sft or grpo

# MemGen configs
TRAIN_METHOD="grpo"   # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

BATCH_SIZE=1

# LOAD_WEAVER_PATH=null
# 直接指向model.safetensors文件
LOAD_WEAVER_PATH='/home/chenhaolong/llmmemory/code/ref/MemGen/results/train/gpqa/Qwen2.5-1.5B-Instruct/pn=8_pl=16_in=5_il=8_20260107-141216/weaver/model.safetensors'

# train
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes ${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active True \
    datasets.mode ${DATASET_MODE} \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method ${TRAIN_METHOD} \
    run.trigger.grpo.per_device_train_batch_size ${BATCH_SIZE} \
    run.trigger.grpo.per_device_eval_batch_size ${BATCH_SIZE} \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.num_generations 8 \
    run.trigger.grpo.gradient_accumulation_steps 4 \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024 \





