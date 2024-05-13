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


def extract_generated_code(resp,  tokenizer=None, sep='Output:\n',):
    if not isinstance(resp, str):
        import vllm  # noqa: 避免全局依赖vllm

    if isinstance(resp, vllm.outputs.RequestOutput):
        # print(resp.prompt[-200:])
        # print('#'*100)
        if getattr(resp, 'prompt', None) is not None:
            resp = resp.prompt + tokenizer.decode(resp.outputs[0].token_ids)
        else:
            resp = tokenizer.decode(resp.prompt_token_ids + resp.outputs[0].token_ids)
        resp = resp.strip()
    ans = resp.split(sep)
    return ans[-1].strip().replace('</s>', '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, help='model dir')
    parser.add_argument('--prompt_corpus_dir', type=str, help='the dir to store prompt corpus')
    parser.add_argument('--run_tasks', type=str, default='EE,NER,RE', help='tasks to run')
    parser.add_argument('--output_file', type=str, default=None, help='output file')
    parser.add_argument('--prediction_name', type=str, default='prediction', help='prediction name')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    args.run_tasks = args.run_tasks.split(',')

    test_task_files = {
        'EE': f'{args.prompt_corpus_dir}/EE/test-prompt.json',
        'NER': f'{args.prompt_corpus_dir}/NER/test-prompt.json',
        'RE': f'{args.prompt_corpus_dir}/RE/test-prompt.json',
    }
    query_settings = [
        'one-stage.zero-shot.prompt--1500_2000',
        'one-stage.first-5-shot.prompt--1500_2000',
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,  # llama的tokenizer似乎存在问题(使用fast的同时需要设置legacy为False)
        legacy=False,
        padding_side='left',  # training时padding在右侧, generation时padding在左侧
        trust_remote_code=True,
    )

    with open(args.output_file, 'rb') as file:
        output_res = pickle.load(file)

    # 5. 提取生成结果至中间文件
    ckpt_extracted_res = {}
    for task_type in args.run_tasks:
        ckpt_extracted_res[task_type] = {}
        for setting in query_settings:
            setting_res = [extract_generated_code(r, tokenizer) for r in output_res[task_type][setting]]
            ckpt_extracted_res[task_type][setting] = setting_res

    ckpt_res_file = os.path.join(output_dir, f'extracted_{args.prediction_name}.json')
    dump_json_file(ckpt_extracted_res, ckpt_res_file, compact=True)
    print(f"extracted results from {args.prediction_name}!")
