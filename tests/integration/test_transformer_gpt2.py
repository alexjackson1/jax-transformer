import sys
import os
import jax

sys.path.append(os.getcwd())

import jax.numpy as jnp
from jaxtyping import Array

import flax.linen as nn

from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Block, GPT2Config

from gpt2 import GPT2_PARAMS, tokenizer, reference_gpt2
from tx import LayerNorm, Embed, PosEmbed, TransformerBlock, Unembed


def test_each_layer():
    tokens: Array = tokenizer("Hello, my name is", return_tensors="jax")["input_ids"]
    batch_size, seq_length = tokens.shape

    init_fn = jax.nn.initializers.normal(stddev=0.02)

    # Embedding
    ## Transformer embedding
    tx_embed_mod = Embed(features=768, num_embeddings=50257)
    tx_embed_params = GPT2_PARAMS["embed"]
    tx_embed_vars = {"params": tx_embed_params}
    tx_embed_out: Array = tx_embed_mod.apply(tx_embed_vars, tokens)

    ## Control embedding
    hf_embed_mod = nn.Embed(50257, 768, embedding_init=init_fn, dtype=jnp.float32)
    hf_embed_params = {"params": reference_gpt2._params["transformer"]["wte"]}
    hf_embed_out: Array = hf_embed_mod.apply(hf_embed_params, tokens)

    ## Compare embeddings
    assert tx_embed_out.shape == hf_embed_out.shape
    assert jnp.allclose(tx_embed_out, hf_embed_out, atol=1e-6, rtol=1e-6)

    # Positional embedding
    ## Transformer positional embedding
    tx_pos_embed_mod = PosEmbed(features=768, num_embeddings=1024)
    tx_pos_embed_params = GPT2_PARAMS["pos_embed"]
    tx_pos_embed_vars = {"params": tx_pos_embed_params}
    tx_pos_embed_out: Array = tx_pos_embed_mod.apply(tx_pos_embed_vars, tokens)

    ## Control positional embedding
    hf_pos_embed_mod = nn.Embed(1024, 768, embedding_init=init_fn, dtype=jnp.float32)
    hf_pos_embed_params = {"params": reference_gpt2._params["transformer"]["wpe"]}
    hf_pos_ids = jnp.broadcast_to(
        jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
    )
    hf_pos_embed_out: Array = hf_pos_embed_mod.apply(hf_pos_embed_params, hf_pos_ids)

    ## Compare positional embeddings
    assert tx_pos_embed_out.shape == hf_pos_embed_out.shape
    assert jnp.allclose(tx_pos_embed_out, hf_pos_embed_out, atol=1e-6, rtol=1e-6)

    # Transformer blocks
    tx_next_input = tx_embed_out + tx_pos_embed_out
    hf_next_input = hf_embed_out + hf_pos_embed_out
    for i in range(12):
        ## Transformer block
        tx_block_mod = TransformerBlock(
            num_heads=12, head_dim=64, model_dim=768, mlp_dim=3072
        )
        tx_block_params = GPT2_PARAMS[f"block_{i}"]
        tx_block_vars = {"params": tx_block_params}
        tx_block_out: Array = tx_block_mod.apply(tx_block_vars, tx_next_input)

        ## Control block
        hf_block_mod = FlaxGPT2Block(GPT2Config.from_pretrained("gpt2"))
        hf_block_params = {"params": reference_gpt2._params["transformer"]["h"][f"{i}"]}
        hf_block_out: Array = hf_block_mod.apply(hf_block_params, hf_next_input)[0]

        ## Compare blocks
        assert tx_block_out.shape == hf_block_out.shape
        assert jnp.allclose(tx_block_out, hf_block_out, atol=1e-2, rtol=1e-2)

        ## Update inputs
        tx_next_input = tx_block_out
        hf_next_input = hf_block_out

    # Layer norm
    ## Transformer layer norm
    tx_ln_f_mod = LayerNorm(epsilon=1e-5)
    tx_ln_f_params = GPT2_PARAMS["ln_f"]
    tx_ln_f_vars = {"params": tx_ln_f_params}
    tx_ln_f_out: Array = tx_ln_f_mod.apply(tx_ln_f_vars, tx_next_input)

    ## Control layer norm
    hf_ln_f_mod = nn.LayerNorm(epsilon=1e-5)
    hf_ln_f_params = {"params": reference_gpt2._params["transformer"]["ln_f"]}
    hf_ln_f_out: Array = hf_ln_f_mod.apply(
        hf_ln_f_params, tx_next_input
    )  # hf_next_input

    ## Compare layer norms
    assert tx_ln_f_out.shape == hf_ln_f_out.shape
    assert jnp.allclose(tx_ln_f_out, hf_ln_f_out, atol=1e-6, rtol=1e-6)

    # Unembedding
    ## Transformer unembedding
    tx_unembed_mod = Unembed(features=768, num_embeddings=50257)
    tx_unembed_params = GPT2_PARAMS["unembed"]
    tx_unembed_vars = {"params": tx_unembed_params}
    tx_unembed_out: Array = tx_unembed_mod.apply(tx_unembed_vars, tx_ln_f_out)

    ## Control unembedding
    hf_unembed_mod = nn.Dense(
        50257,
        768,
        kernel_init=init_fn,
        bias_init=nn.initializers.zeros,
        dtype=jnp.float32,
        precision=None,
    )
    hf_unembed_params = {
        "params": {
            "kernel": jnp.transpose(
                reference_gpt2._params["transformer"]["wte"]["embedding"]
            ),
            "bias": jnp.zeros((50257,)),
        }
    }
    hf_unembed_out: Array = hf_unembed_mod.apply(hf_unembed_params, hf_ln_f_out)

    ## Compare unembeddings
    assert tx_unembed_out.shape == hf_unembed_out.shape
    assert jnp.allclose(tx_unembed_out, hf_unembed_out, atol=1e-6, rtol=1e-6)
