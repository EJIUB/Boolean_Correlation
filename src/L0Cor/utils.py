import numpy as np
import random

def Build_Design(n: int, p1: int, k: int, p2: int, seed: int, rho: float):
    """
    Build a design matrix X from:
      - X_active: n x k base active variables (Uniform[0, 1])
      - X1:       n x p1 correlated expansions of the k active vars (AR(1) columns per active var)
      - X2:       n x p2 columns as the mean over random subsets of active variables

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n : int
        Number of samples (rows).
    p1 : int
        Number of columns to generate in X1 (distributed across k active variables).
    k : int
        Number of active variables (columns in X_active).
    p2 : int
        Number of columns to generate in X2.
    rho: float
        correlation between varialbes. 

    Returns
    -------
    X : (n, p1 + p2) ndarray
        Concatenation of X1 and X2.
    X1 : (n, p1) ndarray
        AR(1) expansions built around X_active columns.
    X2 : (n, p2) ndarray
        Means of random subsets of X_active columns (subset size >= 1).
    X_active : (n, k) ndarray
        Base active variables.
    active_idx : list[int]
        Column indices within X1 where each active variable’s seed column appears (first column of each part).
        These indices are relative to X1 (and to X as well, since X1 is at the front).
    """
    # --- config ---
    rho = rho  # AR(1) correlation parameter; change here if desired
    assert n >= 0 and p1 >= 0 and p2 >= 0 and k >= 0, "n, p1, p2, k must be non-negative integers"

    rng = np.random.default_rng(seed)

    # --- X_active ---
    # Use the same RNG for reproducibility (instead of np.random.uniform)
    X_active = rng.uniform(0.0, 1.0, size=(n, k))

    # --- X1: distribute p1 across k parts exactly ---
    parts = []
    active_idx = []
    X1_col_offset = 0

    if p1 > 0 and k > 0:
        base = p1 // k          # minimum columns per active variable
        rem = p1 % k            # first `rem` active variables get +1 column

        for h in range(k):
            part_len = base + (1 if h < rem else 0)
            if part_len == 0:
                continue

            # Initialize part; first column is the active variable itself
            X1_part = np.empty((n, part_len), dtype=float)
            X1_part[:, 0] = X_active[:, h]
            active_idx.append(X1_col_offset)

            # AR(1) expansion for remaining columns
            # x_t = rho * x_{t-1} + sqrt(1 - rho^2) * noise
            for t in range(1, part_len):
                noise = rng.normal(loc=0.0, scale=1.0, size=n)
                X1_part[:, t] = rho * X1_part[:, t - 1] + np.sqrt(1.0 - rho**2) * noise

            parts.append(X1_part)
            X1_col_offset += part_len

        X1 = np.concatenate(parts, axis=1) if parts else np.empty((n, 0), dtype=float)
    else:
        X1 = np.empty((n, 0), dtype=float)

    # --- X2: mean over random subsets of X_active (ensure subset size >= 1) ---
    X2 = np.empty((n, p2), dtype=float)
    if p2 > 0:
        if k == 0:
            # No active variables to draw from; fill zeros
            X2[:, :] = 0.0
        else:
            for t in range(p2):
                m = rng.integers(1, k + 1)            # subset size in [1, k]
                sID = rng.integers(0, k, size=m)      # indices in [0, k-1], with replacement
                X2[:, t] = X_active[:, sID].mean(axis=1)
    # else: X2 stays with shape (n, 0) if p2 == 0

    # --- final concatenate ---
    X = np.concatenate([X1, X2], axis=1) if (p1 + p2) > 0 else np.empty((n, 0), dtype=float)
    
    return X, active_idx

