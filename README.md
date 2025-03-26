# Transformer from Scratch (JAX/Flax Implementation)

A re-implementation of Callum McDougall's "Transformer from Scratch" using JAX and Flax.

## Overview

This repository contains a clean, modular implementation of the Transformer architecture using JAX and Flax. The implementation follows the original architecture described in the "Attention Is All You Need" paper, with a focus on readability and educational value.

## Structure

- `config.py` - Configuration dataclass with model hyperparameters
- `modules.py` - Core transformer components:
  - `LayerNorm` - Layer normalisation
  - `Embed` - Token embedding layer
  - `PosEmbed` - Positional embedding layer
  - `Attention` - Multi-head self-attention mechanism
  - `MLP` - Multi-layer perceptron with GELU activation
  - `TransformerBlock` - Complete transformer block with attention, MLP and residual connections
  - `Unembed` - Final projection layer to vocabulary
- `transformer.py` - Full transformer model
- `tests/` - Test suite for all modules
- `transformer.ipynb` - Jupyter notebook for experimentation

## Model Architecture

The implementation includes a standard transformer architecture with:

- Token embeddings + positional embeddings
- Multiple transformer blocks, each containing:
  - Multi-head self-attention with causal masking
  - Layer normalisation (applied before attention and MLP)
  - MLP with GELU activation
  - Residual connections
- Final layer normalisation and projection to vocabulary

## Default Configuration

The default model configuration is:
- Hidden dimension (`d_model`): 768
- Attention heads (`n_heads`): 12
- Layers (`n_layers`): 12
- Context length (`n_ctx`): 1024
- MLP dimension (`d_mlp`): 3072
- Vocabulary size (`d_vocab`): 50257

## Requirements

- JAX
- Flax
- Jaxtyping
- Einops

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install jax flax jaxtyping einops`
3. Run tests: `python -m tests.test_modules`
4. Experiment with the model in the Jupyter notebook: `jupyter notebook transformer.ipynb`

## Usage Example

```python
import jax.numpy as jnp
import jax.random as jr

from config import Config
from transformer import Transformer

# Initialize model with default config
cfg = Config()
model = Transformer(cfg=cfg)

# Generate random input tokens
tokens = jnp.ones((2, 4), jnp.int32)  # Batch size 2, sequence length 4
rng = jr.PRNGKey(0)

# Initialize parameters
variables = model.init(rng, tokens)

# Run forward pass
logits = model.apply(variables, tokens)
```

## Acknowledgements

This implementation is based on Callum McDougall's "Transformer from Scratch" and adapted to use JAX and Flax. It aims to provide a clear, educational implementation of the transformer architecture.

This work was supported by UK Research and Innovation [grant number EP/S023356/1], in the UKRI Centre for Doctoral Training in Safe and Trusted Artificial Intelligence ([safeandtrustedai.org](https://safeandtrustedai.org)).
