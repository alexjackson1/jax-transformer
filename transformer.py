import flax.linen as nn
from jaxtyping import Array, Float, Int

from config import Config
from modules import Embed, LayerNorm, PosEmbed, TransformerBlock, Unembed


class Transformer(nn.Module):
    cfg: Config

    def setup(self):
        self.embed = Embed(self.cfg)
        self.pos_embed = PosEmbed(self.cfg)
        self.blocks = [TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        self.unembed = Unembed(self.cfg)
        self.norm = LayerNorm(self.cfg)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p v"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.unembed(x)
        return x
