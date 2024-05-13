import os
import json
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import torch


PARAM_MAP = {
    "7B": {
        "n_layers": 32,
    },
    "13B": {
        "n_layers": 40,
    },
    "30B": {
        "n_layers": 60,
    },
    "65B": {
        "n_layers": 80,
    },
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, ori_path):
    assert model_size in PARAM_MAP
    os.makedirs(model_path, exist_ok=True)

    params = PARAM_MAP[model_size]
    n_layers = params["n_layers"]

    loaded = {}
    ORIGINAL_TOKENIZER_SIZE = 32000
    for pt in tqdm(Path(input_base_path).iterdir()):
        # assert tp/mp == 1
        sd = torch.load(pt, map_location="cpu")
        if not pt.name.startswith('layer_'):
            continue
        if pt.name == 'layer_00-model_00-model_states.pt':
            # loaded['model.embed_tokens.weight'] = sd['weight'][: ORIGINAL_TOKENIZER_SIZE, :]
            loaded['model.embed_tokens.weight'] = sd['weight']
            continue
        if pt.name == f'layer_{n_layers + 1}-model_00-model_states.pt':
            loaded['model.norm.weight'] = sd['weight']
            continue
        if pt.name == f'layer_{n_layers + 2}-model_00-model_states.pt':
            # loaded['lm_head.weight'] = sd['weight'][: ORIGINAL_TOKENIZER_SIZE, :]
            loaded['lm_head.weight'] = sd['weight']
            continue

        layer_i = int(pt.name.split('-')[0].replace('layer_', '')) - 1
        layer_sd = { f"model.layers.{layer_i}.{nm}": weight for nm, weight in sd.items() }
        loaded.update(layer_sd)


    torch.save(loaded, os.path.join(model_path, "pytorch_model.bin"))
    
    shutil.copy(f'{ori_path}/config.json', os.path.join(model_path, "config.json"))
    shutil.copy(f'{ori_path}/generation_config.json', os.path.join(model_path, "generation_config.json"))
    shutil.copy(f'{ori_path}/special_tokens_map.json', os.path.join(model_path, "special_tokens_map.json"))
    shutil.copy(f'{ori_path}/tokenizer_config.json', os.path.join(model_path, "tokenizer_config.json"))
    shutil.copy(f'{ori_path}/tokenizer.model', os.path.join(model_path, "tokenizer.model"))

def process_file(pt, n_layers):
    if not pt.name.startswith('layer_'):
        return None
    sd = torch.load(pt, map_location="cpu")
    result = {}
    if pt.name == 'layer_00-model_00-model_states.pt':
        result['model.embed_tokens.weight'] = sd['weight']
    elif pt.name == f'layer_{n_layers + 1}-model_00-model_states.pt':
        result['model.norm.weight'] = sd['weight']
    elif pt.name == f'layer_{n_layers + 2}-model_00-model_states.pt':
        result['lm_head.weight'] = sd['weight']
    else:
        layer_i = int(pt.name.split('-')[0].replace('layer_', '')) - 1
        layer_sd = {f"model.layers.{layer_i}.{nm}": weight for nm, weight in sd.items()}
        result.update(layer_sd)
    return result

def write_model_v2(model_path, input_base_path, model_size, ori_path):
    assert model_size in PARAM_MAP
    os.makedirs(model_path, exist_ok=True)

    params = PARAM_MAP[model_size]
    n_layers = params["n_layers"]

    loaded = {}
    input_path = Path(input_base_path)

    files = [input_path for input_path in input_path.iterdir() if input_path.name.startswith('layer_')]
    print(f"start to convert: {os.path.basename(input_path)}")
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_file, files, [n_layers] * len(files)))
        executor.shutdown(wait=True)
        for result in results:
            if result is not None:
                loaded.update(result)

    torch.save(loaded, os.path.join(model_path, "pytorch_model.bin"))
    
    for file_name in ["config.json", "generation_config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.model"]:
        shutil.copy(os.path.join(ori_path, file_name), os.path.join(model_path, file_name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--ori_model_dir",
    )
    args = parser.parse_args()
    write_model_v2(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        ori_path=args.ori_model_dir
    )

def copy_config():
    for i in range(200, 1400, 200):
        model_path = f"../outputs/llm/llama2_13b_wiki_2048/global_step{i}_hf"
        ori_path = "../pretrain_models/LLaMA-2/llama-2-13b-hf"
        shutil.copy(f'{ori_path}/config.json', os.path.join(model_path, "config.json"))
        shutil.copy(f'{ori_path}/generation_config.json', os.path.join(model_path, "generation_config.json"))
        shutil.copy(f'{ori_path}/special_tokens_map.json', os.path.join(model_path, "special_tokens_map.json"))
        shutil.copy(f'{ori_path}/tokenizer_config.json', os.path.join(model_path, "tokenizer_config.json"))
        shutil.copy(f'{ori_path}/tokenizer.model', os.path.join(model_path, "tokenizer.model"))

if __name__ == "__main__":
    main()
    # copy_config()
