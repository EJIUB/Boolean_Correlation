import time
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def tune_elastic_net_once(X, y, cv=5, seed=1, l1_ratio_grid=(0.1,0.5,0.9,0.95,1.0), max_iter=20000):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    model = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=l1_ratio_grid, cv=cv, random_state=seed, n_jobs=-1, max_iter=max_iter))
    model.fit(X, y)
    en = model.named_steps["elasticnetcv"]
    return {"alpha": float(en.alpha_), "l1_ratio": float(en.l1_ratio_)}

def solve_elastic_net_fixed(X, y, alpha, l1_ratio, seed=1, max_iter=20000):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    model = make_pipeline(StandardScaler(), ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=max_iter))
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    beta = model.named_steps["elasticnet"].coef_
    sel = np.flatnonzero(beta != 0).astype(int)
    return {"beta": beta, "selected": sel, "meta": {"runtime_sec": dt, "alpha": float(alpha), "l1_ratio": float(l1_ratio)}}