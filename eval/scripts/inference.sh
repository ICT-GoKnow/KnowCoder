#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_512_scx6592

module load compilers/cuda/11.8 anaconda/2021.11 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x
export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64:/home/bingxing2/apps/compilers/cuda/cuda-11.8_dev230901:$LD_LIBRARY_PATH
source activate KnowCoder_sft_env 
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=True
export OMP_NUM_THREADS=1

JOB_ID="${SLURM_JOB_ID}"

date +"%Y-%m-%d %H:%M:%S"

ckpts=${1:-"10"}

run_tasks=${2:-"NER"}

experiment_name=test

benchmark_dir=../corpus/benchmark/Knowcoder-Fewshot-Benchmark

output_dir=outputs/$experiment_name

log_dir=$output_dir/logs

if [ ! -d ${log_dir} ];then
    mkdir ${log_dir}
fi

benchmark_name=${benchmark_dir##*/}
prediction_dir=$output_dir/lora_merged_results/$benchmark_name

IFS=","

for ckpt in ${ckpts}
do
    echo "run evaluation for ${ckpt} ..."
    python src/inference.py \
        --model_dir $output_dir/lora_merged/sft_ckpt_${ckpt} \
        --prompt_corpus_dir $benchmark_dir \
        --run_tasks "$run_tasks" \
        --output_file $prediction_dir/ckpt_${ckpt}_res.pkl \
        --prediction_name "sft_ckpt_${ckpt}" >> $log_dir/eval_${ckpt}_${JOB_ID}.log 2>&1
done

date +"%Y-%m-%d %H:%M:%S"

shell_dir=$output_dir/shells
if [ ! -d ${shell_dir} ];then
    mkdir ${shell_dir}
fi

cp ${BASH_SOURCE[0]} $shell_dir/eval_${JOB_ID}.sh

# sbatch -p vip_gpu_512_scx6592 scripts/eval.sh
