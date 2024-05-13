#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_json_file(file):
    with open(file, 'r', encoding='UTF-8') as file:
        data = json.load(file)
    return data


def read_jsonl_file(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def dump_json_file(obj, file, compact=False):
    with open(file, 'w', encoding='UTF-8') as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
        else:
            json.dump(obj, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_corpus_dir', type=str, help='the dir to store prompt corpus')
    parser.add_argument('--ckpts', type=str, help='involved checkpoints')
    parser.add_argument('--run_tasks', type=str, default='EE,NER,RE', help='tasks to run')
    parser.add_argument('--model_type', type=str, default='llama2-7b', help='model type')
    parser.add_argument('--output_dir', type=str, help='output dir')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.run_tasks = args.run_tasks.split(',') if ',' in args.run_tasks else args.run_tasks.split(' ')
    args.ckpts = args.ckpts.split(',') if ',' in args.ckpts else args.ckpts.split(' ')

    test_task_files = {
        'EE': f'{args.prompt_corpus_dir}/EE/test-prompt.json',
        'NER': f'{args.prompt_corpus_dir}/NER/test-prompt.json',
        'RE': f'{args.prompt_corpus_dir}/RE/test-prompt.json',
    }
    query_settings = [
        'one-stage.zero-shot.prompt--1500_2000',
        'one-stage.first-5-shot.prompt--1500_2000',
    ]

    all_ckpt_res = {}
    for ckpt in args.ckpts:
        ckpt_res_file = os.path.join(args.output_dir, f"extracted_{args.model_type}_sft_ckpt_{ckpt}.json")
        all_ckpt_res[ckpt] = read_json_file(ckpt_res_file)
    
    all_settings = set()
    for ckpt in args.ckpts:
        for task_type in args.run_tasks:
            for s in all_ckpt_res[ckpt][task_type].keys():
                all_settings.add(s)
    assert all([s in all_settings for s in query_settings]), "the settings must in the results"
            
    task_sample_len = {}
    for task_type in args.run_tasks:
        if task_type not in task_sample_len:
            task_sample_len[task_type] = set()
        for ckpt in args.ckpts:
            for setting in query_settings:
                task_sample_len[task_type].add(len(all_ckpt_res[ckpt][task_type][setting]))
        assert len(task_sample_len[task_type]) == 1, "different results must have same sample_num"

    for task_type in args.run_tasks:
        # 中间文件名: intermediate_{task_type}.json
        mid_file = os.path.join(args.output_dir, f'intermediate_{task_type}.json')
        if os.path.exists(mid_file):
            cur_data = read_json_file(mid_file)
        else:
            cur_data = read_jsonl_file(test_task_files[task_type])

        assert len(cur_data) == list(task_sample_len[task_type])[0], "results and input file must have same length"

        for ckpt in args.ckpts:
            for setting in query_settings:
                prediction_name = f"{args.model_type}_sft_ckpt_{ckpt}-{setting}"
                print(f'add field-{prediction_name} to {mid_file}')
                for idx in tqdm(range(len(cur_data))):
                    cur_data[idx][prediction_name] = all_ckpt_res[ckpt][task_type][setting][idx]

        dump_json_file(cur_data, mid_file)