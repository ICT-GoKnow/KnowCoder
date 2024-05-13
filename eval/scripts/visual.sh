
tmp_dir="test/one-stage.zero-shot.prompt--1500_2000"

output_dir="test/one-stage.zero-shot.prompt--1500_2000"

ckpts=${1:-"1000,2000,3000,4000,5000,6000,7000,7554"}

python_command=python

run_tasks=${2:-"NER,RE,EE,EAE,EE-1datasets"}

run_match_typs=${3:-"EM"}
eval_granularities="overall"

IFS=","

${python_command} src/visualization.py  \
    --run_tasks "${run_tasks}" \
    --run_match_types "${run_match_typs}" \
    --ckpts "${ckpts}" \
    --intermediate_data_dir "${tmp_dir}" \
    --output_dir "${output_dir}" \
    --granularity "${eval_granularities}"
