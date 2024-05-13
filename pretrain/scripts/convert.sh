#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:1

module load anaconda compilers/cuda/11.6 compilers/gcc/10.3.1 cudnn/8.4.0.27_cuda11.x 
source activate knowcoder_pretrain_env

python convert2ckpt.py --mp_world_size 8 \
    --model_name_or_path ../pretrain_models/LLaMA-2/llama-2-13b-hf \
    --output_dir ../pretrain_models/LLaMA-2/llama-2-13b-hf-ckpt
