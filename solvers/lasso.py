import time
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def tune_lasso_once(X, y, cv=5, seed=1, max_iter=10000):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    model = make_pipeline(StandardScaler(), LassoCV(cv=cv, random_state=seed, n_jobs=-1, max_iter=max_iter))
    model.fit(X, y)
    alpha = float(model.named_steps["lassocv"].alpha_)
    return {"alpha": alpha}

def solve_lasso_fixed(X, y, alpha, seed=1, max_iter=10000):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    model = make_pipeline(StandardScaler(), Lasso(alpha=float(alpha), max_iter=max_iter))
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    beta = model.named_steps["lasso"].coef_
    sel = np.flatnonzero(beta != 0).astype(int)
    return {"beta": beta, "selected": sel, "meta": {"runtime_sec": dt, "alpha": float(alpha)}}