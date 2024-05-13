

data_dir="input"

tmp_dir="output"

ontology_dir="../corpus/onto_data"

eval_type="file"  

python_env=python


ckpts=${1:-"8344"}

tasks=${2:-"EAE"}

match_types=${3:-"EM"}

eval_granularities=${4:-"source"}

experiment_sets=${6:-"one-stage.zero-shot.prompt--1500_2000"}

filter_outlier=1 

wikidata_upper=1 

schema_type="aligned"   

summary=1  


declare -A name_map
name_map=([EAE]="Event" [RE]="Relation" [NER]="Entity" [EE]="Event")
IFS=","


for experiment_set in ${experiment_sets}
do
    for ckpt in ${ckpts}
    do
        if [ $eval_type = 'file' ];then
            echo "start to convert checkpoint-${ckpt} result!"
            if [[ ${tasks} =~ "EAE" ]]; then
                ${python_env} src/convert/convert_eae.py \
                    --input_file ${data_dir}/intermediate_EAE.json \
                    --output_dir ${tmp_dir}/${experiment_set}/EAE/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --filter_outlier ${filter_outlier}\
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            if [[ ${tasks} =~ "NER" ]]; then
                ${python_env} src/convert/convert_ner.py \
                    --input_file ${data_dir}/intermediate_NER.json \
                    --output_dir  ${tmp_dir}/${experiment_set}/NER/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --wikidata_upper ${wikidata_upper} \
                    --filter_outlier ${filter_outlier} \
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            if [[ ${tasks} =~ "RE" ]]; then
                ${python_env} src/convert/convert_re.py \
                    --input_file ${data_dir}/intermediate_RE.json \
                    --output_dir ${tmp_dir}/${experiment_set}/RE/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --filter_outlier ${filter_outlier}\
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            if [[ ${tasks} =~ "EE" ]]; then
                ${python_env} src/convert/convert_ee.py \
                    --input_file ${data_dir}/intermediate_EE.json \
                    --output_dir ${tmp_dir}/${experiment_set}/EE/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --filter_outlier ${filter_outlier}\
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            echo "start to evaluate checkpoint-${ckpt} result!"
            for task_type in ${tasks}
            do
                for match_type in ${match_types}
                do
                    for eval_granularity in ${eval_granularities}
                    do
                        echo "start to evaluate ${task_type} with ${match_type} and ${eval_granularity}!"
                        ${python_env} src/eval.py \
                            --input_file ${data_dir}/intermediate_${task_type}.json \
                            --pred_file ${tmp_dir}/${experiment_set}/${task_type}/${ckpt}/prediction.json \
                            --gold_file ${tmp_dir}/${experiment_set}/${task_type}/${ckpt}/label.json \
                            --task_type ${task_type} \
                            --match_type ${match_type} \
                            --output_dir ${tmp_dir}/${experiment_set}/${task_type}/${ckpt} \
                            --result_file result_${eval_granularity}_${match_type}.json \
                            --granularity ${eval_granularity}
                    done
                done
            done
        else
            echo "start to evaluate checkpoint-${ckpt} result!"
            for task_type in ${tasks}
            do
                for match_type in ${match_types}
                do
                    for eval_granularity in ${eval_granularities}
                    do
                        echo "start to evaluate ${task_type} with ${match_type} and ${eval_granularity}!"
                        ${python_env} src/eval.py \
                            --input_file ${data_dir}/intermediate_${task_type}.json \
                            --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                            --eval_type ${eval_type} \
                            --task_type ${task_type} \
                            --match_type ${match_type} \
                            --output_dir ${tmp_dir}/${experiment_set}/${task_type}/${ckpt} \
                            --result_file result_${eval_granularity}_${match_type}.json \
                            --granularity ${eval_granularity} \
                            --wikidata_upper ${wikidata_upper} \
                            --filter_outlier ${filter_outlier}\
                            --ontology_dir ${ontology_dir} \
                            --schema_type ${schema_type}
                    done
                done
            done
        fi
    done
done


