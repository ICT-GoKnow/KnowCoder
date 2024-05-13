import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings

warnings.filterwarnings('ignore')
import pickle
import os 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from generate_prompt import generate_prompt
import json
from utils import read_json_file, dump_json_file, extract_generated_code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='sft_ckpt_7000', help='model dir')
    args = parser.parse_args()
    if args.model_dir is None:
        raise ValueError('model_dir is None')
    print(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,use_fast=True,legacy=False,padding_side='left',trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    max_num_batched_tokens = 16000
    llm = LLM(model=args.model_dir,tokenizer_mode='auto',trust_remote_code=True,max_num_batched_tokens=max_num_batched_tokens)
    
    sentence=input("Please input a sentence:")
    task=input("Pleas input a task(NER, RE, ED):")
    types=input("Pleas input the types (split with \",\"):")
    types=types.split(",")
    selcted_types={}
    text=[]
    if task=="NER":
        selcted_types=dict({ "entity": types, "relation": [], "ed": [], "eae": [] })
        entity_prompt, relation_prompt, ed_prompt, eae_prompt = generate_prompt(selcted_types, sentence, 'results = []')
        text = [entity_prompt]
    if task=="RE":
        selcted_types=dict({ "entity": [], "relation": types, "ed": [], "eae": [] })
        entity_prompt, relation_prompt, ed_prompt, eae_prompt = generate_prompt(selcted_types, sentence, 'results = []')
        text = [relation_prompt]
    if task=="ED":
        selcted_types=dict({ "entity": [], "relation": [], "ed": types, "eae": [] })
        entity_prompt, relation_prompt, ed_prompt, eae_prompt = generate_prompt(selcted_types, sentence, 'results = []')
        text = [ed_prompt]
    sampling_params = SamplingParams(n=1,best_of=1,temperature=0,stop='"""',max_tokens=512 + 128)
    output=llm.generate(text,sampling_params)
    output1=extract_generated_code(output[0], tokenizer)
    print("Output:"+output1)
    #output2=extract_generated_code(output[1], tokenizer)
    #print("A:"+output2)
    #output3=extract_generated_code(output[2], tokenizer)
    #print("A:"+output3)
#
