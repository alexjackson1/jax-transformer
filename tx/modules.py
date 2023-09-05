from functools import partial
from typing import Iterable

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import flax.linen as nn
import einops


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
    init_range: float = 0.02

    def setup(self):
        shape = (self.num_embeddings, self.features)
        init_fn = nn.initializers.normal(self.init_range)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        return jnp.take(self.embedding, tokens, axis=0)


class PosEmbed(nn.Module):
    num_embeddings: int
    features: int
    init_range: float = 0.02

    def setup(self):
        shape = (self.num_embeddings, self.features)
        init_fn = nn.initializers.normal(self.init_range)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.embedding[:seq_len], "p m -> b p m", b=batch)


class Attention(nn.Module):
    num_heads: int
    head_dim: int
    model_dim: int
    init_range: float = 0.02

    def setup(self):
        self.causal_mask = nn.make_causal_mask(
            jnp.ones((1, 1024), dtype="bool"), dtype="bool"
        )

        init_dense = partial(
            nn.DenseGeneral,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
        )

        self.c_attn = init_dense(features=3 * self.model_dim)
        self.c_proj = init_dense(features=self.model_dim)

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        batch_size = x.shape[0]

        hidden_states = self.c_attn(x)
        query, key, value = self._split_outputs(hidden_states)
        query_length, key_length = query.shape[1], key.shape[1]

        causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        mask_shape = (batch_size,) + causal_mask.shape[1:]
        causal_mask = jnp.broadcast_to(causal_mask, mask_shape)

        attention_bias = lax.select(
            causal_mask > 0,
            jnp.full(causal_mask.shape, 0.0).astype(jnp.float32),
            jnp.full(causal_mask.shape, jnp.finfo(jnp.float32).min).astype(jnp.float32),
        )

        attn_weights = nn.attention.dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=None,
            deterministic=True,
            dtype=jnp.float32,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)

        return attn_output

    def _split_outputs(self, hidden_states: Array):
        return map(self._split_heads, jnp.split(hidden_states, 3, axis=2))

    def _split_heads(self, hidden_states: Array):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states: Array):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.model_dim,))


class MLP(nn.Module):
    features: Iterable[int]
    init_range: float = 0.02

    def setup(self):
        dense_init = partial(
            nn.DenseGeneral,
            axis=-1,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
        )
        self.fc_1 = dense_init(features=self.features[0])
        self.proj = dense_init(features=self.features[1])

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x = self.fc_1(x)
        x = nn.gelu(x)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    model_dim: int
    mlp_dim: int
    epsilon: float = 1e-5
    init_range: float = 0.02

    def setup(self):
        self.ln_1 = LayerNorm(epsilon=self.epsilon)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            model_dim=self.model_dim,
            init_range=self.init_range,
        )
        self.ln_2 = LayerNorm(epsilon=self.epsilon)
        self.mlp = MLP(
            features=[self.mlp_dim, self.model_dim],
            init_range=self.init_range,
        )

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x_norm = self.ln_1(x)
        x = self.attn(x_norm) + x
        x_norm = self.ln_2(x)
        x = self.mlp(x_norm) + x
        return x


class Unembed(nn.Module):
    features: int
    num_embeddings: int
    init_range: float = 0.02

    def setup(self):
        init_fn = jax.nn.initializers.normal(stddev=self.init_range)
        shape = (self.features, self.num_embeddings)
        self.kernel = self.param("kernel", init_fn, shape)
        self.bias = self.param("bias", jax.nn.initializers.zeros, (shape[-1],))

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p v"]:
        x = einops.einsum(x, self.kernel, "b p m, m v -> b p v")
        x = x + self.bias
        return x
