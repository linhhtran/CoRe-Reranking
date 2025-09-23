from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import DynamicCache
import torch

class DynamicCacheWithQuery(DynamicCache):
    def __init__(self, query_indices=[]) -> None:
        super().__init__()
        self._query_indices = query_indices # indices for query vectors to save
        self.query_cache = []

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if query_states is not None:
            if len(self.query_cache) <= layer_idx:
                self.query_cache.append(query_states)
            else:
                self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(None, key_states, value_states, layer_idx)
        return cache