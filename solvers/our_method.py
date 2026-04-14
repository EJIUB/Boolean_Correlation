import time
import numpy as np
from sklearn.model_selection import KFold
from scipy.linalg import cho_factor, cho_solve

from src.L0Cor.algo import Run_PQN, ProjCappedSimplex_NIPS21_Fix_LinearSearch, ProjCSimplex_Gurobi


def _our_obj(u, X, y, H, pho, mu):
    """
    Objective and gradient used by PQN.

    u: (p,) or (p,1)
    X: (n,p)
    y: (n,)
    H: (p,p) correlation-like matrix
    pho, mu: scalars
    """
    u = np.asarray(u, dtype=float).reshape(-1, 1)  # (p,1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)  # (n,1)

    # Xu = X * u.T (broadcast across columns)
    Xu = X * u.T  # (n,p)

    # A = (Xu @ X.T)/pho + I
    A = (Xu @ X.T) / float(pho)
    A.flat[:: A.shape[0] + 1] += 1.0

    # Solve A^{-1} y via Cholesky
    c, lower = cho_factor(A, overwrite_a=True, check_finite=False)
    My = cho_solve((c, lower), y, check_finite=False)

    Hu = H @ u
    XMy = X.T @ My

    f = np.float64((y.T @ My)[0, 0] + float(mu) * (u.T @ Hu)[0, 0])
    g = (-(XMy**2) / float(pho) + 2.0 * float(mu) * Hu).astype(np.float64)
    return f, g


def _ols_refit_beta(X, y, selected):
    """Refit OLS on selected indices and return full-length beta (p,)."""
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    selected = np.asarray(selected, dtype=int)

    if selected.size == 0:
        return beta

    XS = X[:, selected]
    coef, *_ = np.linalg.lstsq(XS, y, rcond=None)
    beta[selected] = coef
    return beta


# ---------- Solve (fixed hyperparams) ----------
def solve_our_method_fixed(
    X,
    y,
    k_select: int,
    mu: float,
    pho: float,
    max_iter: int = 500,
    verbose: int = 0,
    seed: int = 1,
):
    """
    Runs our method once with fixed hyperparameters.
    Returns dict(beta, selected, meta) matching your solver contract.
    """
    np.random.seed(int(seed))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, p = X.shape
    k_select = min(int(k_select), p)

    # Correlation matrix (same as notebook)
    Cor = np.corrcoef(X, rowvar=False)
    col_var = np.var(X, axis=0, ddof=1)  # sample variance
    std = np.sqrt(col_var)
    H = Cor*Cor#np.abs(Cor * np.outer(std, std))
    #H = np.corrcoef(X, rowvar=False)
    

    funObj = lambda w: _our_obj(w, X, y, H, pho=float(pho), mu=float(mu))
    funProj = lambda w: ProjCappedSimplex_NIPS21_Fix_LinearSearch(w, k_select)

    u0 = np.ones((p, 1), dtype=float) / float(p)

    t0 = time.perf_counter()
    uout, obj, info = Run_PQN(funObj, funProj, u0, max_iter=int(max_iter), verbose=int(verbose))
    dt = time.perf_counter() - t0

    u = np.asarray(uout, dtype=float).reshape(-1)
    selected = np.argsort(-u)[:k_select].astype(int)

    beta = _ols_refit_beta(X, y, selected)

    return {
        "beta": beta,
        "selected": selected,
        "meta": {
            "runtime_sec": dt,
            "mu": float(mu),
            "pho": float(pho),
            "k_select": int(k_select),
            "max_iter": int(max_iter),
        },
    }


# ---------- Tune once (CV over a small grid) ----------
def tune_our_method_once(
    X,
    y,
    k_select: int,
    mu_grid=(5.0, 20.0, 50.0),
    pho_grid=(0.2, 1.0),
    max_iter: int = 500,
    cv: int = 3,
    seed: int = 1,
):
    """
    Tune mu/pho once using CV MSE computed after OLS refit on selected set.
    Returns a dict of best hyperparameters to cache for runs 1..9.
    """
    np.random.seed(int(seed))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    kf = KFold(n_splits=int(cv), shuffle=True, random_state=int(seed))

    best = None
    best_score = float("inf")

    for mu in mu_grid:
        for pho in pho_grid:
            fold_mse = []
            for tr, te in kf.split(X):
                out = solve_our_method_fixed(
                    X[tr],
                    y[tr],
                    k_select=k_select,
                    mu=float(mu),
                    pho=float(pho),
                    max_iter=max_iter,
                    verbose=0,
                    seed=seed,
                )
                beta = out["beta"]
                pred = X[te] @ beta
                fold_mse.append(float(np.mean((y[te] - pred) ** 2)))

            score = float(np.mean(fold_mse))
            if score < best_score:
                best_score = score
                best = {
                    "k_select": int(k_select),
                    "mu": float(mu),
                    "pho": float(pho),
                    "max_iter": int(max_iter),
                }

    return best