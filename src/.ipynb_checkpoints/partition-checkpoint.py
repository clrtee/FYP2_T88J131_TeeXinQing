# src/partition.py
import numpy as np
from typing import List


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.1,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Non-IID partition by Dirichlet distribution over labels.
    Returns: list of index arrays, one per client.

    - Smaller alpha => more heterogeneous (more non-IID)
    - Larger alpha => closer to IID
    """
    rng = np.random.default_rng(seed)

    y = np.asarray(y)
    n = len(y)
    all_idxs = np.arange(n)

    # group indices by class
    classes = np.unique(y)
    class_indices = {c: all_idxs[y == c] for c in classes}

    # prepare buckets for each client
    client_idxs = [[] for _ in range(num_clients)]

    # allocate per-class indices using Dirichlet proportions
    for c in classes:
        idx_c = class_indices[c].copy()
        rng.shuffle(idx_c)

        # sample proportions
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))

        # convert proportions to split sizes
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)
        split_points[-1] = len(idx_c)  # ensure exact end

        prev = 0
        for k in range(num_clients):
            take = idx_c[prev:split_points[k]]
            client_idxs[k].extend(take.tolist())
            prev = split_points[k]

    # shuffle within each client
    out = []
    for k in range(num_clients):
        idxs = np.array(client_idxs[k], dtype=np.int64)
        rng.shuffle(idxs)
        out.append(idxs)
    return out


def iid_partition(
    y: np.ndarray,
    num_clients: int,
    seed: int = 42
) -> List[np.ndarray]:
    """
    True IID split: shuffle all sample indices, then evenly split into num_clients parts.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return [p.astype(np.int64) for p in parts]
