#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_512_scx6592

module load compilers/cuda/11.8 anaconda/2021.11 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x
export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64:/home/bingxing2/apps/compilers/cuda/cuda-11.8_dev230901:$LD_LIBRARY_PATH
source activate knowcoder_sft_env
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=True
export OMP_NUM_THREADS=1

JOB_ID="10000"

date +"%Y-%m-%d %H:%M:%S"

ckpts=${1:-"10"}

experiment_name=test

model_name_or_path=../pretrain_models/LLaMA-2/Llama-2-7b-hf

template=KnowCoder
output_dir=outputs/$experiment_name
IFS=","

log_dir=$output_dir/logs

if [ ! -d ${log_dir} ];then  
    mkdir ${log_dir}
fi

for ckpt in ${ckpts}
do
    python src/export_model.py \
        --model_name_or_path $model_name_or_path \
        --template $template \
        --finetuning_type lora \
        --checkpoint_dir $output_dir/checkpoint-${ckpt} \
        --export_dir $output_dir/lora_merged/sft_ckpt_${ckpt} >> $log_dir/merge_${ckpt}_${JOB_ID}.log 2>&1
done

date +"%Y-%m-%d %H:%M:%S"

shell_dir=$output_dir/shells
if [ ! -d ${shell_dir} ];then
    mkdir ${shell_dir}
fi

cp ${BASH_SOURCE[0]} $shell_dir/merge_lora_ckpts_${JOB_ID}.sh

# sbatch -p vip_gpu_512_scx6592 scripts/merge_lora_ckpts.sh
