import json
import subprocess
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _ols_beta(X, y, selected):
    n, p = X.shape
    beta = np.zeros(p, dtype=float)

    selected = np.atleast_1d(selected).astype(int)
    if selected.size == 0:
        return beta

    XS = X[:, selected]
    coef, *_ = np.linalg.lstsq(XS, y, rcond=None)
    beta[selected] = coef
    return beta


def _call_holp_R(X, y, num_select, family="gaussian", rscript="Rscript"):
    """
    Calls R screening::screening(method='holp') and returns selected indices (0-based).
    Caps num_select so it never exceeds p.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, p = X.shape

    # CAP num_select to [1, p]
    ns = int(num_select)
    ns = max(1, min(ns, p))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x_csv, y_csv, out_json = td / "X.csv", td / "y.csv", td / "out.json"
        pd.DataFrame(X).to_csv(x_csv, index=False, header=False)
        pd.DataFrame(y).to_csv(y_csv, index=False, header=False)

        cmd = [
            rscript, "src/screening/run_holp.R",
            str(x_csv), str(y_csv), str(out_json),
            str(ns), str(family)
        ]

        t0 = time.perf_counter()
        try:
            # capture stderr so errors are actionable
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            msg = f"HOLP R call failed (num_select={ns}, p={p}).\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            raise RuntimeError(msg) from e
        dt = time.perf_counter() - t0

        res = json.loads(out_json.read_text())
        sel = np.array(res.get("selected_1based", []), dtype=int) - 1
        sel = sel[sel >= 0]  # safety
        return sel.astype(int), {"runtime_sec": float(dt), "num_select": int(ns), "family": family}


def tune_holp_once(X, y, cv=3, seed=1, num_select_grid=None):
    """
    Tune num_select once using CV MSE after OLS refit on HOLP-selected set.
    Ensures num_select never exceeds p.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, p = X.shape

    if num_select_grid is None:
        # Safe default grid that scales with n but never exceeds p
        candidates = sorted(set([
            max(1, min(p, k)) for k in [
                10,
                20,
                30,
                int(n / 4),
                int(n / 2),
                int(3 * n / 4),
                p
            ]
        ]))
        num_select_grid = candidates
    else:
        # sanitize user grid
        num_select_grid = sorted(set(int(max(1, min(p, ns))) for ns in num_select_grid))

    kf = KFold(n_splits=int(cv), shuffle=True, random_state=int(seed))

    best_ns = None
    best_score = float("inf")

    for ns in num_select_grid:
        fold_mse = []
        for tr, te in kf.split(X):
            sel, _ = _call_holp_R(X[tr], y[tr], num_select=ns)
            beta = _ols_beta(X[tr], y[tr], sel)
            pred = X[te] @ beta
            fold_mse.append(float(np.mean((y[te] - pred) ** 2)))
        score = float(np.mean(fold_mse))

        if score < best_score:
            best_score = score
            best_ns = int(ns)

    return {"num_select": int(best_ns)}


def solve_holp_fixed(X, y, num_select, family="gaussian", seed=1):
    """
    Solve with fixed num_select.
    Returns beta from OLS refit for consistent top-k ranking.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    sel, meta = _call_holp_R(X, y, num_select=num_select, family=family)
    beta = _ols_beta(X, y, sel)
    return {"beta": beta, "selected": sel, "meta": meta}