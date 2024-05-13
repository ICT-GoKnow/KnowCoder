#!/bin/bash
module load compilers/cuda/11.8 anaconda/2021.11 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x
export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64:/home/bingxing2/apps/compilers/cuda/cuda-11.8_dev230901:$LD_LIBRARY_PATH
source activate knowcoder_sft_env 
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=True
export OMP_NUM_THREADS=1

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

NODES=1
NUM_GPUS=4
NODE_RANK=1
MASTER_ADDR=1
MASTER_PORT=29501
JOB_ID=10000

log_dir=logs
if [ ! -d ${log_dir} ];then
    mkdir ${log_dir}
fi

shell_dir=shells
if [ ! -d ${shell_dir} ];then
    mkdir ${shell_dir}
fi

cp ${BASH_SOURCE[0]} $shell_dir/train_rank${NODE_RANK}_${JOB_ID}.sh

#logs
OUTPUT_LOG="$log_dir/train_rank${NODE_RANK}_${JOB_ID}.log"
echo "nodes,gpus,mp_size,node_rank,master_addr,master_port,dhostfile" >> $OUTPUT_LOG
echo "$NODES,$NUM_GPUS,$MP_SIZE,$NODE_RANK,$MASTER_ADDR,$MASTER_PORT,$DHOSTFILE" >> $OUTPUT_LOG

dataset=example_data
output_dir=outputs/example_experiment

ds_config=configs/deepspeed/sft/ds_config.json
model_name_or_path=../pretrain_models/LLaMA-2/Llama-2-7b-hf

template=KnowCoder
date +"%Y-%m-%d %H:%M:%S"

torchrun --nnodes ${NODES} \
	--nproc_per_node ${NUM_GPUS} \
	--node_rank=${NODE_RANK} \
	--master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train_bash.py \
    --deepspeed ${ds_config} \
    --stage sft \
    --do_train \
    --model_name_or_path ${model_name_or_path} \
    --template ${template} \
    --dataset ${dataset} \
    --output_dir ${output_dir} \
    --lora_target gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --finetuning_type lora \
    --lora_target all \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 100 \
    --learning_rate 3e-4 \
    --num_train_epochs 1.0 \
    --lora_rank 32 \
    --warmup_ratio 0.1 \
    --plot_loss \
    --fp16 \
    --flash_attn \
    --seed 42 \
    --ddp_timeout 18000 \
    --dataloader_num_workers 1 \
    --cutoff_len 2048 >> $OUTPUT_LOG 2>&1

date +"%Y-%m-%d %H:%M:%S"

cp $ds_config $output_dir/ds_config.json
cp $shell_dir/train_rank${NODE_RANK}_${JOB_ID}.sh $output_dir/train_rank${NODE_RANK}_${JOB_ID}.sh

cp slurm-${JOB_ID}.out $output_dir/slurm-${JOB_ID}.out
cp $log_dir/train_rank${NODE_RANK}_${JOB_ID}.log $output_dir/train_rank${NODE_RANK}_${JOB_ID}.log

# --overwrite_cache
# --save_strategy epoch
