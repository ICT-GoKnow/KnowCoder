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


def dump_json_file(obj, file):
    with open(file, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def dump_jsonl_file(records, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        for record in records:
            outline = json.dumps(record, ensure_ascii=False)
            outfile.write(outline + "\n")


def extract_generated_code(resp,  tokenizer=None, sep='Output:\n',):
    if not isinstance(resp, str):
        import vllm  

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


def build_prompt(query, resp='', system=None, sep='\n'):
    if system is None:
        system=(
            "You are a highly skilled assistant at digesting "
            "and extracting information from textual content. "
            "Below is an input containing standard type definitions "
            "and textual content. Please complete it with the "
            "extracted information in the form of structured code."
        )
    pattern = "Input:\n{query}\nOutput:\n{resp}"
    if system:
        return sep.join([system, pattern.format(query=query, resp=resp)])
    else:
        return pattern.format(query=query, resp=resp)


def build_input(tokenizer, query, resp='', system=None, sep='\n', head_max_len=1500, tail_max_len=2000):
    prompt = build_prompt(query=query, resp=resp, system=system, sep=sep)
    token_ids = tokenizer.encode(prompt)
    if head_max_len is not None and tail_max_len is not None and head_max_len + tail_max_len < len(token_ids):
        return token_ids[:head_max_len] + token_ids[-tail_max_len:]
    return token_ids


def retrieve_dict(dic, key_with_sep, sep='.'):
    level_key = key_with_sep.split(sep)
    val = dic
    for key in level_key:
        val = val[key]
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, help='model dir')
    parser.add_argument('--prompt_corpus_dir', type=str,
                        default='/workspace/user_code/work/factgpt/corpus/prompt_construction',
                        help='the dir to store prompt corpus')
    parser.add_argument('--run_tasks', type=str, default='EE,NER,RE', help='tasks to run')
    parser.add_argument('--output_file', type=str, default=None, help='output file')
    parser.add_argument('--prediction_name', type=str, default='prediction', help='prediction name')
    args = parser.parse_args()

    if args.model_dir is None:
        raise ValueError('model_dir is None')

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    args.run_tasks = args.run_tasks.split(',')

    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,  
        legacy=False,
        padding_side='left',  
        trust_remote_code=True,
    )
    # model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

    
    test_task_files = {
        'ED': f'{args.prompt_corpus_dir}/ED/test-prompt.json',
        'NER': f'{args.prompt_corpus_dir}/NER/test-prompt.json',
        'RE': f'{args.prompt_corpus_dir}/RE/test-prompt.json',
        'EAE': f'{args.prompt_corpus_dir}/EAE/test-prompt.json',
        'ED-1dataset': f'{args.prompt_corpus_dir}/ED-1dataset/test-prompt.json',
        'EAE-1dataset': f'{args.prompt_corpus_dir}/EAE-1dataset/test-prompt.json',
    }

    
    query_settings = [
        
        #'one-stage.zero-shot.prompt--1500_2000',
        'one-stage.first-5-shot.prompt--1500_2000',
    ]

    all_prompts = {}
    for task_type in args.run_tasks:
        test_data_file = test_task_files[task_type]
        all_prompts[task_type] = {}
        test_data = read_jsonl_file(test_data_file)
        for it in tqdm(test_data):
            for setting in query_settings:
                if setting not in all_prompts[task_type]:
                    all_prompts[task_type][setting] = []
                if '--' in setting:
                    key, _max_len = setting.split('--')
                    head_max_len, tail_max_len = [int(x) for x in _max_len.split('_')]
                else:
                    key, head_max_len, tail_max_len = setting, None, None
                query = retrieve_dict(it, key)
                token_ids = build_input(tokenizer=tokenizer, query=query,
                                        head_max_len=head_max_len, tail_max_len=tail_max_len)
                all_prompts[task_type][setting].append(token_ids)

    max_num_seqs = 4
    max_model_len = 4096
    max_num_batched_tokens = max_num_seqs * max_model_len
    llm = LLM(model=args.model_dir,
              tokenizer_mode='auto',
              trust_remote_code=True,
              max_num_seqs=max_num_seqs,
              max_model_len=max_model_len,
              max_num_batched_tokens=max_num_batched_tokens)
    print('model loaded!!!!')

    
    max_out_len = 512 + 128 
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        temperature=0.0,
        stop='"""',
        max_tokens=max_out_len,
    )

  
    print(f'run_tasks: {args.run_tasks}')
    output_res = {}
    for task_type in args.run_tasks:
        output_res[task_type] = {}
        for setting in query_settings:
            print(f'processing: {task_type}-{setting} ...')
            output_res[task_type][setting] = [''] * len(all_prompts[task_type][setting])
            good_indices = [idx for idx, seq in enumerate(all_prompts[task_type][setting]) if len(seq) < max_model_len]
            good_len, bad_len = len(good_indices), len(all_prompts[task_type][setting]) - len(good_indices)
            print(f'good_samples={good_len}, bad_samples={bad_len}, all_samples={len(all_prompts[task_type][setting])}')
            
            good_samples = [all_prompts[task_type][setting][idx] for idx in good_indices]
            good_sample_results = llm.generate(prompt_token_ids=good_samples, sampling_params=sampling_params)
            
            assert len(good_indices) == len(good_sample_results)
            for idx, r in zip(good_indices, good_sample_results):
                output_res[task_type][setting][idx] = r

   
    with open(args.output_file, 'wb') as file:
        pickle.dump(output_res, file)


    for task_type in args.run_tasks:
        
        task_res = {}
        for setting in query_settings:
            setting_res = [extract_generated_code(r, tokenizer) for r in output_res[task_type][setting]]
            task_res[setting] = setting_res

        assert len(set([len(v) for v in task_res.values()])) == 1, 

       
        mid_file = f'{output_dir}/intermediate_{task_type}.json'
        if os.path.exists(mid_file):
            cur_data = read_json_file(mid_file)
        else:
            cur_data = read_jsonl_file(test_task_files[task_type])

        assert len(cur_data) == len(task_res[query_settings[0]]),

        for idx in tqdm(range(len(cur_data))):
            for setting in query_settings:
                cur_data[idx][args.prediction_name + '-' + setting] = task_res[setting][idx]

        dump_json_file(cur_data, mid_file)
        print(f'add field-{args.prediction_name} to {mid_file}')
