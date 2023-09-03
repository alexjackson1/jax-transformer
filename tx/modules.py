import functools
from typing import Iterable

import jax
import jax.numpy as jnp

import flax.linen as nn
from jaxtyping import Array, Float, Int

import einops


# Array shape abbreviations:
# b: batch (batch size)
# p: position (sequence length)
# m: model dimension (embedding size)
# v: token index (vocab size)


class LayerNorm(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - x_mean) / jnp.sqrt(x_var + self.epsilon)

        scale = self.param("scale", jax.nn.initializers.ones, (x.shape[-1],))
        bias = self.param("bias", jax.nn.initializers.zeros, (x.shape[-1],))
        x = x * scale + bias

        return x


class Embed(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        shape = (self.num_embeddings, self.features)
        embedding = self.param("embedding", jax.nn.initializers.normal(0.02), shape)
        return jnp.take(embedding, tokens, axis=0)


class PosEmbed(nn.Module):
    context_length: int
    features: int

    @nn.compact
    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        shape = (self.context_length, self.features)
        embedding = self.param("embedding", jax.nn.initializers.normal(0.02), shape)
        batch, seq_len = tokens.shape
        return einops.repeat(embedding[:seq_len], "p m -> b p m", b=batch)


class Attention(nn.Module):
    num_heads: int
    head_dim: int
    model_dim: int

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        kernel_init = jax.nn.initializers.normal(0.02)
        bias_init = jax.nn.initializers.zeros

        kernel_shape = (self.num_heads, self.model_dim, self.head_dim)
        bias_shape = (self.num_heads, self.head_dim)

        # Query linear transformation
        kernel_query = self.param(f"kernel_query", kernel_init, kernel_shape)
        bias_query = self.param(f"bias_query", bias_init, bias_shape)
        query = einops.einsum(x, kernel_query, "b p m, n m h -> b p n h") + bias_query

        # Key linear transformation
        kernel_key = self.param(f"kernel_key", kernel_init, kernel_shape)
        bias_key = self.param(f"bias_key", bias_init, bias_shape)
        key = einops.einsum(x, kernel_key, "b p m, n m h -> b p n h") + bias_key

        # Value linear transformation
        kernel_value = self.param(f"kernel_value", kernel_init, kernel_shape)
        bias_value = self.param(f"bias_value", bias_init, bias_shape)
        value = einops.einsum(x, kernel_value, "b p m, n m h -> b p n h") + bias_value

        # Calculate attention scores
        attn_scores = einops.einsum(query, key, "b pq n h, b pk n h -> b n pq pk")
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.head_dim**0.5)
        attn_pattern = jax.nn.softmax(attn_scores_masked, axis=-1)

        # Compute weighted average of values (by applying attention pattern)
        z = einops.einsum(value, attn_pattern, "b pk n h, b n pq pk -> b pq n h")

        # Calculate output (by applying kernel and summing over heads, then adding bias)
        out_shape = (self.num_heads, self.head_dim, self.model_dim)
        kernel_out = self.param(f"kernel_out", kernel_init, out_shape)
        bias_out = self.param(f"bias_out", bias_init, (self.model_dim,))
        attn_out = einops.einsum(z, kernel_out, "b pq n h, n h m -> b pq m") + bias_out

        return attn_out

    def apply_causal_mask(self, attn_scores: jax.Array):
        mask = jnp.triu(jnp.ones_like(attn_scores), k=1)
        return attn_scores * mask - 1e10 * (1 - mask)


class MLP(nn.Module):
    features: Iterable[int]
    init_range: float = 0.02

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        layer = functools.partial(
            nn.DenseGeneral,
            axis=-1,
            kernel_init=jax.nn.initializers.normal(self.init_range),
            bias_init=jax.nn.initializers.zeros,
        )

        for i, f in enumerate(self.features):
            x = layer(features=f)(x)
            if i < len(self.features) - 1:
                x = jax.nn.gelu(x)

        return x


class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    model_dim: int
    mlp_dim: int
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x_1 = LayerNorm(epsilon=self.epsilon)(x)
        x = (
            Attention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                model_dim=self.model_dim,
            )(x_1)
            + x
        )

        # MLP
        x_2 = LayerNorm(epsilon=self.epsilon)(x)
        x = MLP(features=[self.mlp_dim, self.model_dim])(x_2) + x

        return x


class Unembed(nn.Module):
    features: int
    num_embeddings: int

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p v"]:
        shape = (self.features, self.num_embeddings)
        scale = self.param("kernel", jax.nn.initializers.normal(0.02), shape)
        x = einops.einsum(x, scale, "b p m, m v -> b p v")

        bias = self.param("bias", jax.nn.initializers.zeros, (self.num_embeddings,))
        x = x + bias

        return x
