import sys
import os

sys.path.append(os.getcwd())

import jax.numpy as jnp
from jaxtyping import Array

from tx import (
    LayerNorm,
    Embed,
    PosEmbed,
    Attention,
    MLP,
    TransformerBlock,
    Unembed,
    Transformer,
)

from gpt2 import GPT2_PARAMS


def test_layer_norm():
    model = LayerNorm()
    variables = {"params": GPT2_PARAMS["ln_f"]}
    input_data = jnp.ones((1, 1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 768)


def test_embed():
    model = Embed(features=768, num_embeddings=50257)
    variables = {"params": GPT2_PARAMS["embed"]}
    input_data = jnp.ones((1, 1024), dtype=jnp.int32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 768)


def test_pos_embed():
    model = PosEmbed(features=768, num_embeddings=1024)
    variables = {"params": GPT2_PARAMS["pos_embed"]}
    input_data = jnp.ones((1, 1024), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 768)


def test_attention():
    model = Attention(num_heads=12, head_dim=64, model_dim=768)
    variables = {"params": GPT2_PARAMS["block_0"]["attn"]}
    input_data = jnp.ones((1, 1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 768)


def test_mlp():
    model = MLP(features=[3072, 768])
    variables = {"params": GPT2_PARAMS["block_0"]["mlp"]}
    input_data = jnp.ones((1, 1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 768)


def test_transformer_block():
    model = TransformerBlock(num_heads=12, head_dim=64, model_dim=768, mlp_dim=3072)
    variables = {"params": GPT2_PARAMS["block_0"]}
    input_data = jnp.ones((1, 1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 768)


def test_unembed():
    model = Unembed(features=768, num_embeddings=50257)
    variables = {"params": GPT2_PARAMS["unembed"]}
    input_data = jnp.ones((1, 1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 50257)


def test_transformer():
    model = Transformer()
    variables = {"params": GPT2_PARAMS}
    input_data = jnp.ones((1, 1024), dtype=jnp.int32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1, 1024, 50257)
