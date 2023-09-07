import dataclasses
from functools import partial
from typing import Callable, Iterable, Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import flax.linen as nn
import einops

from tx.caching import KeyValueCache


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

    def hook(self, module: nn.Module, fn: Callable, label: str = None):
        name = module.name if label is None else f"{module.name}_{label}"
        return Hook(name=f"{name}_hook", fn=fn)

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

        hook_fn = lambda x, n: print(n, x.shape)  #
        self.hook_query = self.hook(self.c_attn, fn=hook_fn, label="query")
        self.hook_key = self.hook(self.c_attn, fn=hook_fn, label="key")
        self.hook_value = self.hook(self.c_attn, fn=hook_fn, label="value")
        self.hook_weights = self.hook(self.c_attn, fn=hook_fn, label="weights")

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        batch_size = x.shape[0]

        hidden_states = self.c_attn(x)
        query, key, value = map(self.hook_query, self._split_outputs(hidden_states))
        query_length, key_length = query.shape[1], key.shape[1]

        causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        mask_shape = (batch_size,) + causal_mask.shape[1:]
        causal_mask = jnp.broadcast_to(causal_mask, mask_shape)

        attention_bias = lax.select(
            causal_mask > 0,
            jnp.full(causal_mask.shape, 0.0).astype(jnp.float32),
            jnp.full(causal_mask.shape, jnp.finfo(jnp.float32).min).astype(jnp.float32),
        )

        attn_weights = self.hook_weights(
            nn.attention.dot_product_attention_weights(
                query,
                key,
                bias=attention_bias,
                dropout_rng=None,
                deterministic=True,
                dtype=jnp.float32,
                precision=None,
            )
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


@dataclasses.dataclass
class ModelConfig:
    model_dim: int = 256
    layer_norm_eps: float = 1e-5
    vocab_dim: int = 50257
    context_length: int = 256
    num_heads: int = 4
    head_dim: int = 64
    mlp_dim: int = 1024
    num_layers: int = 2


class Transformer(nn.Module):
    model_dim: int = 768
    layer_norm_eps: Float = 1e-5
    vocab_dim: int = 50257
    context_length: int = 1024
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 3072
    num_layers: int = 12

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(**config.__dict__)

    def setup(self):
        features = self.model_dim
        self.embed = Embed(features=features, num_embeddings=self.vocab_dim)
        self.pos_embed = PosEmbed(features=features, num_embeddings=self.context_length)
        self.blocks = [self._make_block(i) for i in range(self.num_layers)]
        self.ln_f = LayerNorm(epsilon=self.layer_norm_eps)
        self.unembed = Unembed(features=features, num_embeddings=self.vocab_dim)

    def __call__(self, tokens) -> Array:
        embed = self.embed(tokens)  # text embedding
        pos_embed = self.pos_embed(tokens)  # positional embedding

        x = embed + pos_embed  # combine embeddings
        for block in self.blocks:  # loop over layers/blocks
            x = block(x)  # apply attention and mlp

        x = self.ln_f(x)  # apply final layer norm
        logits = self.unembed(x)  # unembed to logits

        return logits

    def _make_block(self, i: int):
        return TransformerBlock(
            name=f"block_{i}",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            model_dim=self.model_dim,
            mlp_dim=self.mlp_dim,
            epsilon=self.layer_norm_eps,
        )


class Hook(nn.Module):
    name: str
    fn: Callable[[Array, str], None]

    @nn.nowrap
    def __call__(self, x):
        self.fn(x, self.name)
        return x


class HookedTransformer(nn.Module):
    model_dim: int = 768
    layer_norm_eps: Float = 1e-5
    vocab_dim: int = 50257
    context_length: int = 1024
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 3072
    num_layers: int = 12

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(**config.__dict__)

    def _compute_offset(self, cache: Optional[KeyValueCache]) -> int:
        return 0 if cache is None else cache[0].past_keys.shape[1]

    def _sanity_check_inputs(self, tokens: Array, cache: Optional[KeyValueCache]):
        if len(tokens.shape) == 1:
            tokens = tokens[None]

        if cache is not None:
            t_batch, t_ctx = tokens.shape
            (c_batch, c_ctx, c_heads, c_head) = cache[0].past_keys.shape
            assert c_batch == t_batch
            assert c_heads == self.num_heads
            assert c_head == self.head_dim
            assert c_ctx == 0 or t_ctx == 1

        return tokens

    def _make_block(self, i: int):
        return TransformerBlock(
            name=f"block_{i}",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            model_dim=self.model_dim,
            mlp_dim=self.mlp_dim,
            epsilon=self.layer_norm_eps,
        )

    def hook(self, module: nn.Module, fn: Callable):
        return Hook(name=f"{module.name}_hook", fn=fn)

    def setup(self):
        hook_fn = lambda x, n: print(n, x.shape)
        self.embed = Embed(num_embeddings=self.vocab_dim, features=self.model_dim)
        self.embed_hook = self.hook(self.embed, fn=hook_fn)

        self.pos_embed = PosEmbed(
            num_embeddings=self.context_length, features=self.model_dim
        )
        self.pos_embed_hook = self.hook(self.pos_embed, fn=hook_fn)

        self.blocks = [self._make_block(i) for i in range(self.num_layers)]
        self.ln_f = LayerNorm(epsilon=self.layer_norm_eps)
        self.unembed = Unembed(num_embeddings=self.vocab_dim, features=self.model_dim)

    def __call__(
        self, tokens: Int[Array, "batch pos"], kv_cache: Optional[KeyValueCache] = None
    ) -> Array:
        tokens = self._sanity_check_inputs(tokens, kv_cache)
        offset = self._compute_offset(kv_cache)

        embed = self.embed_hook(self.embed(tokens))
        pos_embed = self.pos_embed_hook(self.pos_embed(tokens))

        residual = embed + pos_embed
        for i, block in enumerate(self.blocks):
            cache_entry = kv_cache[i] if kv_cache is not None else None
            # residual = block(residual, past_kv_cache_entry=cache_entry)
            residual = block(residual)

        residual = self.ln_f(residual)
        logits = self.unembed(residual)

        return logits
