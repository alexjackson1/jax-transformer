import dataclasses

import jax
import jax.numpy as jnp

import flax.linen as nn
from jaxtyping import Array, Float, Int

import einops

from config import Config


class LayerNorm(nn.Module):
    cfg: Config

    kernel: Float[Array, "m"] = dataclasses.field(init=False)
    bias: Float[Array, "m"] = dataclasses.field(init=False)

    def setup(self):
        shape = (self.cfg.d_model,)
        self.kernel = self.param("kernel", jax.nn.initializers.ones, shape)
        self.bias = self.param("bias", jax.nn.initializers.zeros, shape)

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_std = jnp.std(x, axis=-1, keepdims=True)
        x = (x - x_mean) / x_std
        x = x * self.kernel + self.bias
        return x


class Embed(nn.Module):
    cfg: Config

    embedding: Float[Array, "v m"] = dataclasses.field(init=False)

    def setup(self):
        init_fn = jax.nn.initializers.normal(self.cfg.init_range)
        shape = (self.cfg.d_vocab, self.cfg.d_model)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        return jnp.take(self.embedding, tokens, axis=0)


class PosEmbed(nn.Module):
    cfg: Config

    embedding: Float[Array, "c m"] = dataclasses.field(init=False)

    def setup(self):
        init_fn = jax.nn.initializers.normal(self.cfg.init_range)
        shape = (self.cfg.n_ctx, self.cfg.d_model)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        batch, seq_len = tokens.shape
        shape = (batch, seq_len, self.cfg.d_model)
        return jnp.broadcast_to(self.embedding[:seq_len], shape)


class Attention(nn.Module):
    cfg: Config

    kernel_query: Float[Array, "n m h"] = dataclasses.field(init=False)
    kernel_key: Float[Array, "n m h"] = dataclasses.field(init=False)
    kernel_value: Float[Array, "n m h"] = dataclasses.field(init=False)
    kernel_out: Float[Array, "n m h"] = dataclasses.field(init=False)

    bias_query: Float[Array, "n h"] = dataclasses.field(init=False)
    bias_key: Float[Array, "n h"] = dataclasses.field(init=False)
    bias_value: Float[Array, "n h"] = dataclasses.field(init=False)
    bias_out: Float[Array, "n h"] = dataclasses.field(init=False)

    def setup(self):
        kernel_init_fn = jax.nn.initializers.normal(self.cfg.init_range)
        bias_init_fn = jax.nn.initializers.zeros

        qkv_shape = (self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        out_shape = (self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
        bias_shape = (self.cfg.n_heads, self.cfg.d_head)

        self.kernel_query = self.param(f"kernel_query", kernel_init_fn, qkv_shape)
        self.kernel_key = self.param(f"kernel_key", kernel_init_fn, qkv_shape)
        self.kernel_value = self.param(f"kernel_value", kernel_init_fn, qkv_shape)
        self.kernel_out = self.param(f"kernel_out", kernel_init_fn, out_shape)

        self.bias_query = self.param(f"bias_query", bias_init_fn, bias_shape)
        self.bias_key = self.param(f"bias_key", bias_init_fn, bias_shape)
        self.bias_value = self.param(f"bias_value", bias_init_fn, bias_shape)
        self.bias_out = self.param(f"bias_out", bias_init_fn, (self.cfg.d_model,))

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        # Calculate query, key, and value vectors
        qkv_tx = "b p m, n m h -> b p n h"
        query = einops.einsum(x, self.kernel_query, qkv_tx) + self.bias_query
        key = einops.einsum(x, self.kernel_key, qkv_tx) + self.bias_key
        value = einops.einsum(x, self.kernel_value, qkv_tx) + self.bias_value

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_op = "b pq n h, b pk n h -> b n pq pk"
        attn_scores = einops.einsum(query, key, attn_op)
        attn_scores_scaled = attn_scores / self.cfg.d_head**0.5
        attn_scores_masked = self.apply_causal_mask(attn_scores_scaled)
        attn_pattern = jax.nn.softmax(attn_scores_masked, axis=-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z_op = "b pk n h, b n pq pk -> b pq n h"
        z = einops.einsum(value, attn_pattern, z_op)

        # Calculate output (by applying kernel and summing over heads, then adding bias)
        out_op = "b pq n h, n h m -> b pq m"
        attn_out = einops.einsum(z, self.kernel_out, out_op) + self.bias_out

        return attn_out

    def apply_causal_mask(self, attn_scores: jax.Array):
        mask = jnp.triu(jnp.ones_like(attn_scores), k=1)
        return attn_scores * mask - 1e10 * (1 - mask)


class MLP(nn.Module):
    cfg: Config

    kernel_in: Float[Array, "m mlp"] = dataclasses.field(init=False)
    kernel_out: Float[Array, "mlp m"] = dataclasses.field(init=False)

    bias_in: Float[Array, "mlp"] = dataclasses.field(init=False)
    bias_out: Float[Array, "m"] = dataclasses.field(init=False)

    def setup(self):
        kernel_init_fn = jax.nn.initializers.normal(self.cfg.init_range)
        bias_init_fn = jax.nn.initializers.zeros

        self.kernel_in = self.param(
            f"kernel_in",
            kernel_init_fn,
            (self.cfg.d_model, self.cfg.d_mlp),
        )
        self.kernel_out = self.param(
            f"kernel_out",
            kernel_init_fn,
            (self.cfg.d_mlp, self.cfg.d_model),
        )

        self.bias_in = self.param(f"bias_in", bias_init_fn, (self.cfg.d_mlp,))
        self.bias_out = self.param(f"bias_out", bias_init_fn, (self.cfg.d_model,))

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x = einops.einsum(x, self.kernel_in, "b p m, m mlp -> b p mlp") + self.bias_in
        x = jax.nn.gelu(x)
        x = einops.einsum(x, self.kernel_out, "b p mlp, mlp m -> b p m") + self.bias_out
        return x


class TransformerBlock(nn.Module):
    cfg: Config

    ln1: LayerNorm = dataclasses.field(init=False)
    attn: Attention = dataclasses.field(init=False)
    ln2: LayerNorm = dataclasses.field(init=False)
    mlp: MLP = dataclasses.field(init=False)

    def setup(self):
        self.ln1 = LayerNorm(self.cfg)
        self.attn = Attention(self.cfg)
        self.ln2 = LayerNorm(self.cfg)
        self.mlp = MLP(self.cfg)

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class Unembed(nn.Module):
    cfg: Config

    kernel: Float[Array, "m v"] = dataclasses.field(init=False)
    bias: Float[Array, "v"] = dataclasses.field(init=False)

    def setup(self):
        init_fn = jax.nn.initializers.normal(self.cfg.init_range)
        shape = (self.cfg.d_model, self.cfg.d_vocab)
        self.kernel = self.param("kernel", init_fn, shape)
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.cfg.d_vocab,))

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p v"]:
        x = einops.einsum(x, self.kernel, "b p m, m v -> b p v")
        x = x + self.bias
        return x
