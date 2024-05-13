""" feader.py """

import copy
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union
from collections import defaultdict
import numpy as np

import torch
import deepspeed
import transformers
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

import common.utils as utils
from common.utils import is_rank_0
from common.log import logger_rank0 as logger


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

PROMPT_FIELD = 'prompt'
OUTPUT_FIELD = 'output'


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    input_ids = [tokenizer.encode(text, max_length=1024, return_tensors='pt') for text in strings]
    # input_ids = np.array(input_ids)
    input_ids_lens = labels_lens = [
        len(item) for item in input_ids
    ]
    # input_ids = [input_ids]
    # input_ids_lens = [input_ids_lens]
    print(input_ids_lens)

    labels = copy.deepcopy(input_ids)
    labels_lens = copy.deepcopy(input_ids_lens)

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mode: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    samples = [s + t for s, t in zip(sources, targets)]
    samples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (samples, sources)]
    input_ids = samples_tokenized["input_ids"]
    if mode == "sft":
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            print(type(label), source_len)
            label[:source_len] = IGNORE_INDEX
    elif mode == "pretrain":
        labels = copy.deepcopy(input_ids)
    else:
        raise ValueError('Unvalid training mode.')

    # shift
    return dict(
        input_ids=[ids[: -1] for ids in input_ids],
        labels=[lbs[1: ]for lbs in labels]
    )


class PromptDataset(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, data_path: Union[str, Path], eos: str = ""):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'

        self.samples = []
        all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

        error_count = defaultdict(int)
        ERROR_THRESHOLD = 10
        for single_file in tqdm(all_files, disable=not is_rank_0()):
            with (single_file).open(encoding='utf-8') as f:
                for lnum, ln in enumerate(f):
                    try:
                        sample = json.loads(ln)
                        prompt, output = sample[PROMPT_FIELD], sample[OUTPUT_FIELD]
                        if not isinstance(prompt, str) or not isinstance(output, str):
                            raise ValueError()
                        self.samples.append(dict(
                            prompt=prompt,
                            output=output + eos,
                        ))
                    except:
                        logger.warning(f'{single_file}: {lnum} unvalid.')
                        error_count[str(single_file)] += 1

                    if error_count[str(single_file)] > ERROR_THRESHOLD:
                        logger.warning(f'{single_file} exceeds max error number. skipped.')
                        break

        logger.info(f'total samples num: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]


@dataclass
class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    mode: str

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]

        data_dict = preprocess(sources, targets, self.tokenizer, self.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels
        )


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=42, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_prompt_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, val_split=None) -> Dict:
    assert val_split is None
    dataset = PromptDataset(data_path=data_args.data_path, eos=tokenizer.eos_token)
    if is_rank_0():
        print(f"{len(dataset)} datasets")
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()

    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))


@dataclass
class DataCollatorForLLaMA(object):
    """Collate for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    mode: str
    max_seq_len: int

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [], []
        if self.mode == 'pt':
            texts = [sample['text'] for sample in samples]
            max_length_in_examples = 0
            for idx, text in enumerate(texts):
                # res = self.tokenizer.encode(text)
                # print(res)
                input_id = self.tokenizer.encode(text)[:self.max_seq_len]
                
                # make shift label
                label = copy.deepcopy(input_id)
                label = label[1:] + [self.tokenizer.eos_token_id]
                # pos = random.randint(int(0.2 * len(input_id)) + 1, int(0.7 * len(input_id)))
                # label = [IGNORE_INDEX] * pos + label[pos:]

                if len(input_id) < self.max_seq_len:
                    input_id += [0] * (self.max_seq_len-len(input_id))
                    label += [IGNORE_INDEX] * (self.max_seq_len-len(label))
                # max_length_in_examples = max(max_length_in_examples, len(input_id))
                
                input_ids.append(torch.tensor(input_id))
                labels.append(torch.tensor(label))
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
        elif self.mode == 'sft':
            texts = [(sample['input'], sample['output']) for sample in samples]
            for idx, (input, output) in enumerate(texts):
                # res = self.tokenizer.encode(text)
                # print(res)
                input_id_1 = self.tokenizer.encode(input)[:self.max_seq_len]
                
                left_len = self.max_seq_len-len(input_id_1)
                if left_len > 0:                
                    input_id_2 = self.tokenizer.encode(output)[:self.max_seq_len]
                    input_id_2 = input_id_2[1:]
                    input_id_2 = input_id_2[:left_len]

                    input_id = input_id_1 + input_id_2
                else:
                    input_id = input_id_1
                
                # make shift label
                label = copy.deepcopy(input_id)
                label = label[1:] + [self.tokenizer.eos_token_id]
                pos = len(input_id_1)
                label = [IGNORE_INDEX] * pos + label[pos:]
                assert len(label) == len(input_id), f"{len(label)}, {len(input_id)}"

                if len(input_id) < self.max_seq_len:
                    input_id += [0] * (self.max_seq_len-len(input_id))
                    label += [IGNORE_INDEX] * (self.max_seq_len-len(label))
                
                input_ids.append(torch.tensor(input_id))
                labels.append(torch.tensor(label))
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)

        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels
        )


def make_hot_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, trainer_args):

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path, cache_dir='./cache')
    else:
        data = load_dataset(data_args.data_path, cache_dir='./cache')

    train_data = data["train"]
    data_len = len(train_data)
    data_collator = DataCollatorForLLaMA(tokenizer=tokenizer, mode=data_args.mode, max_seq_len=trainer_args.max_seq_len)
    g = torch.Generator()

    dataloader = DataLoader(train_data,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            generator=g,)

    return (iter(deepspeed.utils.RepeatingLoader(dataloader)), data_len,)
