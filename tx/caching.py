from dataclasses import dataclass
from typing import Iterable, List
from jaxtyping import Array
import jax.numpy as jnp


@dataclass
class KeyValueCacheEntry:
    past_keys: Array
    past_values: Array

    @classmethod
    def init_cache_entry(cls, shape: Iterable[int], dtype: jnp.dtype = jnp.float32):
        return cls(
            past_keys=jnp.empty(shape, dtype=dtype),
            past_values=jnp.empty(shape, dtype=dtype),
        )


@dataclass
class KeyValueCache:
    entries: List[KeyValueCacheEntry]

    @classmethod
    def init_cache(
        cls, layers: int, shape: Iterable[int], dtype: jnp.dtype = jnp.float32
    ):
        entries = [
            KeyValueCacheEntry.init_cache_entry(shape=shape, dtype=dtype)
            for _ in range(layers)
        ]
        return cls(entries=entries)

    def __getitem__(self, index: int) -> KeyValueCacheEntry:
        return self.entries[index]
