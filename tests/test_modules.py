import sys
import os

sys.path.append(os.getcwd())


import jax.random as jr
import jax.numpy as jnp

import flax.linen as nn
from jaxtyping import Array

from config import Config
from modules import (
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
    cfg = Config(debug=True)
    layer = LayerNorm(cfg)
    shape = (2, 4, 768)

    variables = init(layer, shape)

    params = variables["params"]
    assert jnp.all(params["kernel"]) == 1.0
    assert jnp.all(params["bias"]) == 0.0


def test_layer_norm_apply():
    cfg = Config(debug=True)
    layer = LayerNorm(cfg)
    shape = (2, 4, 768)
    variables = init(layer, shape)

    output = apply_float(layer, variables, shape)

    o_mean, o_var = jnp.mean(output, axis=-1), jnp.var(output, axis=-1)
    tolerance = {"atol": 1e-4, "rtol": 1e-3}
    assert jnp.allclose(o_mean, 0.0, **tolerance) == True
    assert jnp.allclose(o_var, 1.0, **tolerance) == True
    assert output.shape == shape


def test_embed_init():
    cfg = Config(debug=True)
    layer = Embed(cfg)
    shape = (2, 4)

    variables = init(layer, shape, jnp.int32)

    embedding = get_param(variables, "embedding")
    assert embedding.shape == (cfg.d_vocab, cfg.d_model)


def test_embed_apply():
    cfg = Config(debug=True)
    layer = Embed(cfg)
    shape = (2, 4)
    variables = init(layer, shape, jnp.int32)

    output = apply_int(layer, variables, shape)

    features = layer.cfg.d_model
    assert output.shape == shape + (features,)


def test_pos_embed_init():
    cfg = Config(debug=True)
    layer = PosEmbed(cfg)
    shape = (2, 4)

    variables = init(layer, shape, jnp.int32)

    embedding = get_param(variables, "embedding")
    assert embedding.shape == (cfg.n_ctx, cfg.d_model)


def test_pos_embed_apply():
    cfg = Config(debug=True)
    layer = PosEmbed(cfg)
    shape = (2, 4)
    variables = init(layer, shape, jnp.int32)

    output = apply_int(layer, variables, shape)

    features = layer.cfg.d_model
    assert output.shape == shape + (features,)


def test_attention_init():
    cfg = Config(debug=True)
    layer = Attention(cfg)
    shape = (2, 4, 768)

    init(layer, shape)


def test_attention_apply():
    cfg = Config(debug=True)
    layer = Attention(cfg)
    shape = (2, 4, 768)
    variables = init(layer, shape)

    output = apply_float(layer, variables, shape)

    assert output.shape == shape


def test_mlp_init():
    cfg = Config(debug=True)
    layer = MLP(cfg)
    shape = (2, 4, 768)

    init(layer, shape)


def test_mlp_apply():
    cfg = Config(debug=True)
    layer = MLP(cfg)
    shape = (2, 4, 768)
    variables = init(layer, shape)

    output = apply_float(layer, variables, shape)

    assert output.shape == shape


def test_transformer_block_init():
    cfg = Config(debug=True)
    layer = TransformerBlock(cfg)
    shape = (2, 4, 768)

    init(layer, shape)


def test_transformer_block_apply():
    cfg = Config(debug=True)
    layer = TransformerBlock(cfg)
    shape = (2, 4, 768)
    variables = init(layer, shape)

    output = apply_float(layer, variables, shape)

    assert output.shape == shape


def test_unembed_init():
    cfg = Config(debug=True)
    layer = Unembed(cfg)
    shape = (2, 4, 768)

    init(layer, shape)


def test_unembed_apply():
    cfg = Config(debug=True)
    layer = Unembed(cfg)
    shape = (2, 4, 768)
    variables = init(layer, shape)

    output = apply_float(layer, variables, shape)

    assert output.shape == (*shape[:-1], cfg.d_vocab)
