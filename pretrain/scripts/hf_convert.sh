#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_512_scx6592

module load anaconda compilers/cuda/11.6 compilers/gcc/10.3.1 cudnn/8.4.0.27_cuda11.x 
source activate knowcoder_pretrain_env

for i in $(seq 50 50 300)
do
    python convert2hf.py \
        --input_dir outputs/test/global_step${i} \
        --output_dir outputs/test/global_step${i}_hf \
        --ori_model_dir ../pretrain_models/LLaMA-2/Llama-2-7b-hf \
        --model_size '7B'
done
