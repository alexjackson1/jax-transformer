from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Int

from transformers import FlaxGPT2LMHeadModel, GPT2TokenizerFast, PreTrainedTokenizerBase


def to_params(model: FlaxGPT2LMHeadModel) -> Dict[str, Array]:
    params = model._params["transformer"]
    blocks = params["h"]
    embedding: Array = params["wte"]["embedding"]

    def _block_params(block):
        mlp_fc = block["mlp"]["c_fc"]
        mlp_proj = block["mlp"]["c_proj"]
        return {
            "ln_1": block["ln_1"],
            "attn": {
                "c_attn": {
                    "kernel": jnp.transpose(block["attn"]["c_attn"]["kernel"]),
                    "bias": block["attn"]["c_attn"]["bias"],
                },
                "c_proj": {
                    "kernel": jnp.transpose(block["attn"]["c_proj"]["kernel"]),
                    "bias": block["attn"]["c_proj"]["bias"],
                },
            },
            "ln_2": block["ln_2"],
            "mlp": {
                "fc_1": {
                    "kernel": jnp.transpose(mlp_fc["kernel"]),
                    "bias": mlp_fc["bias"],
                },
                "proj": {
                    "kernel": jnp.transpose(mlp_proj["kernel"]),
                    "bias": mlp_proj["bias"],
                },
            },
        }

    return {
        "embed": params["wte"],
        "pos_embed": params["wpe"],
        **{f"block_{i}": _block_params(blocks[f"{i}"]) for i in range(len(blocks))},
        "ln_f": params["ln_f"],
        "unembed": {
            "kernel": jnp.transpose(embedding),
            "bias": jnp.zeros((embedding.shape[0],)),
        },
    }


model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

GPT2_PARAMS = to_params(model)

from typing import List, Literal, Union


def config_tokenizer(tokenizer: PreTrainedTokenizerBase):
    """
    Sets the tokenizer to use for this model.
    tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer
    default_padding_side (str): "right" or "left", which side to pad on
    """
    # Add special tokens if they are not already added
    token_map = tokenizer.special_tokens_map
    if "eos_token" not in token_map or token_map["eos_token"] is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if "pad_token" not in token_map or token_map["pad_token"] is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    if "bos_token" not in token_map or token_map["bos_token"] is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})

    tokenizer.padding_side = "right"


def to_tokens(
    tokenizer: PreTrainedTokenizerBase,
    input: Union[str, List[str]],
    prepend_bos: Union[bool, None] = False,
    truncate: bool = True,
    max_length: Union[int, None] = 1024,
) -> Int[Array, "batch pos"]:
    if prepend_bos is not None and prepend_bos:
        if isinstance(input, str):
            input = tokenizer.bos_token + input
        else:
            input = [tokenizer.bos_token + string for string in input]
    tokens = tokenizer(
        input,
        return_tensors="jax",
        padding=True,
        truncation=truncate,
        max_length=max_length if truncate else None,
        add_special_tokens=(
            not (
                len(tokenizer("")["input_ids"]) > 0
                and tokenizer("")["input_ids"][0] == tokenizer.bos_token_id
            )
        ),
    )["input_ids"]
    return tokens


def to_str_tokens(
    tokenizer: PreTrainedTokenizerBase,
    input: Union[
        Int[Array, "pos"],
        Int[Array, "1 pos"],
        list,
    ],
    prepend_bos: Union[bool, None] = False,
    padding_side: Union[Literal["left", "right"], None] = None,
) -> Union[str, List[str]]:
    if isinstance(input, list):
        return list(
            map(
                lambda t: to_str_tokens(tokenizer, t, prepend_bos, padding_side),
                input,
            )
        )
    elif isinstance(input, Array):
        tokens = input
        tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
        if tokens.ndim == 0:
            # Don't pass dimensionless tensor
            tokens = jnp.expand_dims(tokens, axis=0)
        assert (
            tokens.ndim == 1
        ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
    else:
        raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
    str_tokens = tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
    return str_tokens


def to_str(
    tokenizer: PreTrainedTokenizerBase,
    tokens: Union[
        Int[Array, ""],
        Int[Array, "batch pos"],
        Int[Array, "pos"],
        List[Int[Array, "pos"]],
    ],
) -> Union[str, List[str]]:
    """
    Converts a tensor of tokens to a string (if rank 1) or a list of strings (if rank 2).

    Accepts lists of tokens and numpy arrays as inputs too (and converts to tensors internally)
    """
    assert tokenizer is not None, "Cannot use to_string without a tokenizer"

    if not isinstance(tokens, Array):
        # We allow lists to be input
        tokens = jnp.array(tokens)

    # I'm not sure what exactly clean_up_tokenization_spaces does, but if
    # it's set, then tokenization is no longer invertible, and some tokens
    # with a bunch of whitespace get collapsed together
    if len(tokens.shape) == 2:
        return tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
    elif len(tokens.shape) <= 1:
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"Invalid shape passed in: {tokens.shape}")
