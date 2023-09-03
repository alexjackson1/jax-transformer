import sys
import os

sys.path.append(os.getcwd())


import jax.random as jr
import jax.numpy as jnp

import flax.linen as nn
from jaxtyping import Array

from tx.modules import (
    MLP,
    Attention,
    Embed,
    LayerNorm,
    PosEmbed,
    TransformerBlock,
    Unembed,
)

RNG = jr.PRNGKey(0)


def init(module: nn.Module, shape, dtype=jnp.float32):
    return module.init(RNG, jnp.ones(shape, dtype))


def apply_float(module: nn.Module, params, shape) -> Array:
    return module.apply(params, jr.uniform(RNG, shape))


def apply_int(module: nn.Module, params, shape) -> Array:
    return module.apply(params, jr.randint(RNG, shape, 100, 1000))


def get_param(variables, name: str) -> Array:
    return variables["params"][name]


def test_layer_norm_init():
    layer = LayerNorm()
    shape = (2, 4, 768)
    variables = init(layer, shape)
    params = variables["params"]
    assert jnp.all(params["scale"]) == 1.0
    assert jnp.all(params["bias"]) == 0.0


def test_layer_norm_apply():
    layer = LayerNorm()
    shape = (2, 4, 768)
    variables = init(layer, shape)
    output = apply_float(layer, variables, shape)
    o_mean, o_var = jnp.mean(output, axis=-1), jnp.var(output, axis=-1)
    tolerance = {"atol": 1e-4, "rtol": 1e-3}
    assert jnp.allclose(o_mean, 0.0, **tolerance) == True
    assert jnp.allclose(o_var, 1.0, **tolerance) == True
    assert output.shape == shape


def test_embed_init():
    layer = Embed(features=768, num_embeddings=50257)
    shape = (2, 4)
    variables = init(layer, shape, jnp.int32)
    embedding = get_param(variables, "embedding")
    assert embedding.shape == (50257, 768)


def test_embed_apply():
    layer = Embed(features=768, num_embeddings=50257)
    shape = (2, 4)
    variables = init(layer, shape, jnp.int32)
    output = apply_int(layer, variables, shape)
    features = layer.features
    assert output.shape == shape + (features,)


def test_pos_embed_init():
    layer = PosEmbed(features=768, context_length=1024)
    shape = (2, 4)
    variables = init(layer, shape, jnp.int32)
    embedding = get_param(variables, "embedding")
    assert embedding.shape == (1024, 768)


def test_pos_embed_apply():
    layer = PosEmbed(features=768, context_length=1024)
    shape = (2, 4)
    variables = init(layer, shape, jnp.int32)
    output = apply_int(layer, variables, shape)
    features = layer.features
    assert output.shape == shape + (features,)


def test_attention_init():
    layer = Attention(num_heads=12, head_dim=64, model_dim=768)
    shape = (2, 4, 768)
    init(layer, shape)


def test_attention_apply():
    layer = Attention(num_heads=12, head_dim=64, model_dim=768)
    shape = (2, 4, 768)
    variables = init(layer, shape)
    output = apply_float(layer, variables, shape)
    assert output.shape == shape


def test_mlp_init():
    layer = MLP(features=[3072, 768])
    shape = (2, 4, 768)
    init(layer, shape)


def test_mlp_apply():
    layer = MLP(features=[3072, 768])
    shape = (2, 4, 768)
    variables = init(layer, shape)
    output = apply_float(layer, variables, shape)
    assert output.shape == shape


def test_transformer_block_init():
    layer = TransformerBlock(
        num_heads=12,
        head_dim=64,
        model_dim=768,
        mlp_dim=3072,
        epsilon=1e-5,
    )
    shape = (2, 4, 768)
    init(layer, shape)


def test_transformer_block_apply():
    layer = TransformerBlock(
        num_heads=12,
        head_dim=64,
        model_dim=768,
        mlp_dim=3072,
        epsilon=1e-5,
    )
    shape = (2, 4, 768)
    variables = init(layer, shape)
    output = apply_float(layer, variables, shape)
    assert output.shape == shape


def test_unembed_init():
    layer = Unembed(features=768, num_embeddings=50257)
    shape = (2, 4, 768)
    init(layer, shape)


def test_unembed_apply():
    layer = Unembed(features=768, num_embeddings=50257)
    shape = (2, 4, 768)
    variables = init(layer, shape)
    output = apply_float(layer, variables, shape)
    assert output.shape == (*shape[:-1], 50257)
