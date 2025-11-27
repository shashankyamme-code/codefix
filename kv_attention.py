import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class KVCachedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_cache_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    # -------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        batch_size, seq_len, _ = query.shape

        Q = self.q_proj(query)
        K_new = self.k_proj(key)
        V_new = self.v_proj(value)

        # ------------------------------------------
        # Cache handling (correct)
        # ------------------------------------------
        if cache is not None and cache.get("key") is not None:
            cached_k = cache["key"]
            cached_v = cache["value"]
            K = torch.cat([cached_k, K_new], dim=1)
            V = torch.cat([cached_v, V_new], dim=1)
        else:
            K = K_new
            V = V_new

        cache_len = K.shape[1] - seq_len

        # ------------------------------------------
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if use_causal_mask:
            scores = self._apply_causal_mask(scores, seq_len, cache_len)

        attention_weights = F.softmax(scores, dim=-1)

        if self.training:
            attention_weights = self.dropout(attention_weights)

        out = torch.matmul(attention_weights, V)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        new_cache = {
            "key": K.detach(),
            "value": V.detach()
        }

        if new_cache["key"].shape[2] >= self.max_cache_len:
            raise ValueError("Cache exceeded maximum length")

        return out, new_cache

    # -------------------------------------------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    # -------------------------------------------------------------
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq, self.d_model)

    # -------------------------------------------------------------
    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        seq_len: int,
        cache_len: int
    ):
        batch, heads, q_len, total_len = scores.shape
        mask = torch.triu(torch.ones(q_len, total_len, device=scores.device), 1 + cache_len)
        mask = mask.bool()
        scores = scores.masked_fill(mask, float("-inf"))
        return scores

    # -------------------------------------------------------------
    def reset_cache(self) -> Dict[str, Optional[torch.Tensor]]:
        return {"key": None, "value": None}

    # -------------------------------------------------------------
    def get_cache_info(self, cache: Optional[Dict[str, torch.Tensor]]):
        if cache is None or cache["key"] is None:
            return {
                "cache_length": 0,
                "cache_size_mb": 0.0,
                "is_full": False
            }

        key_cache = cache["key"]
        seq = key_cache.shape[2]
        mem = key_cache.numel() * key_cache.element_size() * 2
        return {
            "cache_length": seq,
            "cache_size_mb": round(mem / (1024 * 1024), 2),
            "is_full": seq >= self.max_cache_len
        }


# -------------------------------------------------------------
def compute_position_ids(seq_len: int, cache_len: int) -> torch.Tensor:
    return torch.arange(cache_len, cache_len + seq_len)


# -------------------------------------------------------------
def validate_inputs(query, key, value):
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError("Inputs must be 3D")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch must match")
    if key.shape[1] != value.shape[1]:
        raise ValueError("Key/value length mismatch")


# -------------------------------------------------------------
def create_sample_input(batch, seq, d_model, seed=None):
    if seed: torch.manual_seed(seed)
    return (
        torch.randn(batch, seq, d_model),
        torch.randn(batch, seq, d_model),
        torch.randn(batch, seq, d_model),
    )


# -------------------------------------------------------------
if __name__ == "__main__":
    print("KV-Cached MHA ready.")
