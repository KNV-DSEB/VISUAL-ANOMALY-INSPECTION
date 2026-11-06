"""Memory-efficient k-Center Greedy coreset subsampling for PatchCore (PC-01)."""
from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def greedy_coreset_subsampling(
    embeddings: np.ndarray | torch.Tensor,
    ratio: float = 0.10,
    max_patches: int | None = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select representative patch embeddings via k-Center (Farthest Point) Greedy sampling.

    Operates in O(N * K) time and O(N) auxiliary memory without allocating an N x N matrix.
    Returns `(coreset_embeddings [K, D], selected_indices [K])`.
    """
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    else:
        emb_np = np.asarray(embeddings, dtype=np.float32)

    n_total = emb_np.shape[0]
    if n_total == 0:
        raise ValueError("Cannot subsample an empty embedding bank.")

    target_k = max(int(round(n_total * float(ratio))), 1)
    if max_patches is not None and max_patches > 0:
        target_k = min(target_k, int(max_patches))
    target_k = min(target_k, n_total)

    if target_k == n_total:
        idx = np.arange(n_total, dtype=np.int64)
        return emb_np.copy(), idx

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(emb_np).to(device)

    selected = np.empty(target_k, dtype=np.int64)
    first_idx = int(rng.integers(0, n_total))
    selected[0] = first_idx

    center = x[first_idx : first_idx + 1]
    min_sq_dists = torch.sum((x - center) ** 2, dim=1)

    for i in range(1, target_k):
        next_idx = int(torch.argmax(min_sq_dists).item())
        selected[i] = next_idx
        center = x[next_idx : next_idx + 1]
        sq_dists = torch.sum((x - center) ** 2, dim=1)
        min_sq_dists = torch.minimum(min_sq_dists, sq_dists)
        min_sq_dists[next_idx] = -1.0

    return emb_np[selected].copy(), selected
