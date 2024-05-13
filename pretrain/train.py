
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal
from functools import partial

import torch
import transformers
import numpy as np
import deepspeed
from tqdm import tqdm
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch.utils.tensorboard import SummaryWriter
from models.llama_pipeline_model import get_model
from models.patching import (
    replace_llama_attn_with_flash_attn,
)
from transformers import LlamaTokenizer
from feeder import (
    make_prompt_dataloader,
    make_hot_dataloader
)
from common.utils import jload, is_rank_0
from common.log import logger_rank0 as logger

warnings.filterwarnings("ignore")

@dataclass
class ModelArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=False)

@dataclass
class DeepspeedArguments:
    use_deepspeed: Optional[bool] = field(default=True)
    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    world_size: int = field(default=None)
    seed: int = field(default=42)
    deepspeed_config: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    mode: Literal['sft', 'pt'] = 'pt'
    num_workers: int = field(default=1)


@dataclass
class TrainerArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)


def read_ds_config(config_path):
    config = jload(config_path)
    return config

def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainerArguments, DeepspeedArguments))
    model_args, data_args, trainer_args, ds_args = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    assert ds_args.use_deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    ds_args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(ds_args.local_rank)

    ds_config = read_ds_config(ds_args.deepspeed_config)
    data_args.num_workers = 2 * ds_args.world_size // ds_args.pipe_parallel_size // ds_args.model_parallel_size
    data_args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(ds_args.seed)
    np.random.seed(ds_args.seed)
    torch.manual_seed(ds_args.seed)
    deepspeed.runtime.utils.set_random_seed(ds_args.seed)
    tb_dir = f"{trainer_args.output_dir}/tensorboard"
    tb_writer = SummaryWriter(tb_dir)
    if model_args.use_flash_attn:
        logger.info("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()

    tokenizer = LlamaTokenizer.from_pretrained(model_args.init_ckpt)
        # print_rank_0("Set the eos_token_id and bos_token_id of LLama model tokenizer", log_file, global_rank)
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1


    # tokenizer = 
    model_config = transformers.AutoConfig.from_pretrained(model_args.init_ckpt)

    # dataset
    # train_dataloader = make_prompt_dataloader(tokenizer=tokenizer, data_args=data_args)
    train_dataloader, train_data_len = make_hot_dataloader(tokenizer=tokenizer, data_args=data_args, trainer_args=trainer_args)
    # pipeline model
    model = get_model(model_config, ds_args, activation_checkpointing_config)


    engine, _, _, _ = deepspeed.initialize(
        ds_args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    print(f"zero_optimization: {engine.zero_optimization()}, bfloat16_enabled: {engine.bfloat16_enabled()}")

    # use `convert2ckpt.py`
    engine.load_checkpoint(model_args.init_ckpt, load_module_only=True, load_optimizer_states=False, load_lr_scheduler_states=False)
    
    trainer_args.train_steps = train_data_len // engine.train_batch_size()
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f"Num of parameters: {numel}")
    logger.info(f"train_batch_size: {engine.train_batch_size()}, total_train_step: {trainer_args.train_steps}, examples: {train_data_len}")
    get_tflops_func = partial(get_tflops, numel, engine.train_batch_size(), trainer_args.max_seq_len)

    tflops_records = []

    start = time.time()
    for step in tqdm(range(1, trainer_args.train_steps + 1), total=trainer_args.train_steps, disable=(not is_rank_0())):
        # if is_rank_0 and profile_step % profile_step == 0:
        #     prof.start_profile()
        loss = engine.train_batch(data_iter=train_dataloader)
        if ds_args.local_rank == 0:
            if step % trainer_args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / trainer_args.log_steps
                avg_tflops = get_tflops_func(avg_time)
                tflops_records.append(get_tflops_func(avg_time))
                avg_tflops = sum(tflops_records) / len(tflops_records)
                logger.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s, tflops: {avg_tflops}")
                start = now


        if step % trainer_args.save_steps == 0:
            logger.info(f"Saving at step {step}")
            engine.save_checkpoint(trainer_args.output_dir)


if __name__ == "__main__":
    main()
