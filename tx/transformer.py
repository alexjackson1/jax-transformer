import dataclasses

import flax.linen as nn
from jaxtyping import Float, Array

from tx.modules import Embed, LayerNorm, PosEmbed, TransformerBlock, Unembed


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


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    from debug_utils import print_nested_structure

    m = Transformer()
    v = m.init(jax.random.PRNGKey(0), jnp.ones((1, 1024), dtype=jnp.int32))
    print_nested_structure(v)
