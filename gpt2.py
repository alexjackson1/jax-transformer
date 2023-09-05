from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array

from transformers import FlaxGPT2LMHeadModel, GPT2TokenizerFast


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


reference_gpt2 = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

GPT2_PARAMS = to_params(reference_gpt2)
