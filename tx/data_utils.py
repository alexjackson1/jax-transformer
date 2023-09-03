from typing import Dict, List

import numpy as np
import einops

from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer

from jaxtyping import Array, Int


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_batch = 0
        self.num_samples = len(dataset)
        self.indices = np.arange(self.num_samples)
        self.reset()

    def reset(self):
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.current_batch * self.batch_size >= self.num_samples:
            self.reset()
            raise StopIteration

        start = self.current_batch * self.batch_size
        end = min((self.current_batch + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]
        batch = [self.dataset[i.item()] for i in batch_indices]
        batch = {k: np.stack([b[k] for b in batch]) for k in batch[0]}
        self.current_batch += 1
        return batch

    def __iter__(self):
        return self


def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_ds(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    dataset = keep_single_column(dataset, column_name)

    # Tokenizer assumptions
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id

    if pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize(examples: Dict[str, List[str]]) -> Dict[str, Int[Array, "b s"]]:
        text = examples[column_name]

        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = eos_token.join(text)

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]

        # Tokenize the chunks in parallel. Uses jax in this version
        token_map = tokenizer(chunks, return_tensors="np", padding=True)
        tokens = token_map["input_ids"].flatten()

        # Drop padding tokens
        tokens = tokens[tokens != pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len

        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(tokens, "(b s) -> b s", b=num_batches, s=seq_len)
        if add_bos_token:
            prefix = np.full((num_batches, 1), bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    num_proc = num_proc if not streaming else None
    args = {"batched": True, "num_proc": num_proc, "remove_columns": [column_name]}
    tokenized_dataset = dataset.map(tokenize, **args)
    tokenized_dataset.set_format(type="np", columns=["tokens"])
    return tokenized_dataset
