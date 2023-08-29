import flax.linen as nn
from jaxtyping import Float, Array

from modules import Embed, LayerNorm, PosEmbed, TransformerBlock, Unembed


class Transformer(nn.Module):
    model_dim: int = 768
    layer_norm_eps: Float = 1e-5
    vocab_dim: int = 50257
    context_length: int = 1024
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 3072
    num_layers: int = 12

    @nn.compact
    def __call__(self, tokens) -> Array:
        embedding = Embed(
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
        )(tokens)

        pos_embedding = PosEmbed(
            features=self.model_dim,
            context_length=self.context_length,
        )(tokens)

        x = embedding + pos_embedding

        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                epsilon=self.layer_norm_eps,
            )(x)

        x = LayerNorm(epsilon=self.layer_norm_eps)(x)

        logits = Unembed(
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
        )(x)

        return logits


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    m = Transformer()
    v = m.init(jax.random.PRNGKey(0), jnp.ones((1, 1024), dtype=jnp.int32))
    o = m.apply(v, jnp.ones((1, 1024), dtype=jnp.int32))
    print(o.shape)
