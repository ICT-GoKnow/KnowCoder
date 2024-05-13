#!/bin/bash
module load anaconda compilers/cuda/11.6 compilers/gcc/10.3.1 cudnn/8.4.0.27_cuda11.x 
source activate knowcoder_pretrain_env 

data_file='./data/pt_kelm_re_multi_v8_newschema.json'
cache_dir='./cache'

export PYTHONUNBUFFERED=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO

### nodes gpus rank master_addr job_id
# nodes
NODES=$1

# gpus
NUM_GPUS=$2

# rank
NODE_RANK=$3

# Master addr
MASTER_ADDR=$4
MASTER_PORT=29501

# JOB_ID
JOB_ID=$5

# LOGS
OUTPUT_LOG="train_rank${NODE_RANK}_${JOB_ID}.log"
deepspeed  --master_port 29501 train.py \
    --output_dir outputs/test \
    --init_ckpt  ../pretrain_models/LLaMA-2/llama-2-7b-hf-ckpt \
    --data_path ${data_file} \
    --max_seq_len 2048 \
    --train_steps 500 \
    --eval_steps 10 \
    --save_steps 50 \
    --log_steps 1 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config configs/deepspeed/ds_config_mx2048_zero1.json
