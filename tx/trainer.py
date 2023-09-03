import sys
from typing import Callable, Dict, NamedTuple, Union
import dataclasses

import jax
import jax.numpy as jnp

from datasets import load_dataset
import jax.random as jr

from tqdm import tqdm

import flax.struct as struct
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState as FlaxTrainState

from jaxtyping import Array, Float, Int
import optax

from gpt2 import tokenizer
from tx.transformer import Transformer, ModelConfig
from tx.data_utils import tokenize_ds, DataLoader

Batch = Dict[str, Int[Array, "b s"]]
Variables = FrozenVariableDict


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 10
    max_steps_per_epoch: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2


class TrainState(FlaxTrainState):
    loss: float = struct.field(pytree_node=False, default=jnp.nan)


def make_loss_fn(network: Transformer) -> Callable[[Variables, Batch], Float]:
    def loss_fn(params, batch: Batch) -> Array:
        tokens = batch["tokens"]
        logits = network.apply({"params": params}, tokens)
        shift_logits = logits[:, :-1]
        shift_labels = tokens[:, 1:]
        # one hot
        shift_labels = jax.nn.one_hot(shift_labels, network.vocab_dim)
        loss = optax.softmax_cross_entropy(shift_logits, shift_labels)
        loss = loss.mean()
        return loss

    return loss_fn


class DataLoaders(NamedTuple):
    train: DataLoader
    val: DataLoader


class Trainer:
    network: Transformer
    config: TrainConfig
    loss_fn: Union[Callable[[Variables, Batch], Float], None]

    def __init__(
        self,
        network: Transformer,
        config: TrainConfig,
        context_length: int,
    ):
        self.config = config
        self.context_length = context_length
        lr, wd = self.config.learning_rate, self.config.weight_decay

        self.network = network
        self.loss_fn = make_loss_fn(self.network)
        self.opt = optax.adamw(learning_rate=lr, weight_decay=wd)

    @staticmethod
    def init_network(config: ModelConfig) -> Transformer:
        return Transformer.from_config(config)

    @classmethod
    def create(cls, config: ModelConfig, train_config: TrainConfig) -> "Trainer":
        network = cls.init_network(config)
        return cls(network, train_config, config.context_length)

    def init(self, key: jr.KeyArray) -> FrozenVariableDict:
        example_input = jnp.ones((1, self.context_length), dtype=jnp.int32)
        return self.network.init(key, example_input)

    def train(self, variables: FrozenVariableDict, loaders: DataLoaders) -> TrainState:
        @jax.jit
        def train_step(state: TrainState, batch: Batch) -> TrainState:
            loss, grads = jax.value_and_grad(self.loss_fn)(state.params, batch)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def val_step(state: TrainState, batch: Batch) -> Float[Array, ""]:
            tokens = batch["tokens"]
            logits = self.network.apply({"params": state.params}, tokens)[:, :-1]
            preds = jnp.argmax(logits, axis=-1)
            return jnp.mean(preds == tokens[:, 1:])

        init_state = {
            "params": variables["params"],
            "tx": self.opt,
            "apply_fn": self.network.apply,
        }
        state = TrainState.create(**init_state)

        max_steps = self.config.max_steps_per_epoch
        num_epochs = self.config.epochs
        # accuracy = jnp.nan

        progress_bar = tqdm(total=max_steps * num_epochs)

        for epoch in range(num_epochs):
            for i, batch in enumerate(loaders.train):
                state, loss = train_step(state, batch)
                state = state.replace(loss=loss)

                progress_bar.update()
                # desc = f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}"
                desc = f"Epoch {epoch+1}, loss: {loss:.3f}"
                progress_bar.set_description(desc)

                if i >= max_steps:
                    break

            # preds = jax.vmap(val_step, in_axes=(None, 0))(state, batch)
            # accuracy = jnp.mean(preds)

        # return accuracy
        return state


if __name__ == "__main__":
    config = ModelConfig()
    train_config = TrainConfig(epochs=3, batch_size=8)
    trainer = Trainer.create(config, train_config)

    print("Initialising variables...")
    variables = trainer.init(jr.PRNGKey(0))

    print("Loading dataset...")
    ds = load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    ds_tokens = tokenize_ds(
        ds,
        tokenizer,
        max_length=trainer.context_length,
        num_proc=None,
    )
    ds_split = ds_tokens.train_test_split(test_size=1000)

    ldr_train = DataLoader(ds_split["train"], train_config.batch_size, True)
    ldr_test = DataLoader(ds_split["test"], train_config.batch_size)
    loaders = DataLoaders(ldr_train, ldr_test)

    print("Training...")
    state = trainer.train(variables, loaders)
    print(state.loss)
