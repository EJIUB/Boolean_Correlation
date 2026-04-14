from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional
import numpy as np

from solvers.lasso import tune_lasso_once, solve_lasso_fixed
from solvers.elastic_net import tune_elastic_net_once, solve_elastic_net_fixed
from solvers.our_method import solve_our_method_fixed, tune_our_method_once
from solvers.precision_lasso import tune_precision_lasso_once, solve_precision_lasso_fixed
from solvers.wlasso import tune_wlasso_once, solve_wlasso_fixed
from solvers.holp import tune_holp_once, solve_holp_fixed

# NEW: unified ncvreg wrapper
from solvers.ncvreg_all import tune_ncvreg_once, solve_ncvreg_fixed


@dataclass
class MethodAPI:
    tune_fn: Callable[..., Dict[str, Any]]
    solve_fn: Callable[..., Dict[str, Any]]
    default_tune_kwargs: Dict[str, Any]
    default_solve_kwargs: Dict[str, Any]


# def _topk_from_beta(beta: np.ndarray, k: int) -> np.ndarray:
#     beta = np.asarray(beta, dtype=float).reshape(-1)
#     k = min(int(k), beta.size)
#     return np.argsort(-np.abs(beta))[:k].astype(int)

def _topk_from_beta(beta: np.ndarray, k: int, X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    k = min(int(k), beta.size)

    if np.all(np.abs(beta) < 1e-12) and X is not None and y is not None:
        # fallback: abs correlation with y
        X0 = X - np.mean(X, axis=0, keepdims=True)
        y0 = y - np.mean(y)
        denom = (np.linalg.norm(X0, axis=0) * np.linalg.norm(y0) + 1e-12)
        corr = np.abs((X0.T @ y0) / denom)
        return np.argsort(-corr)[:k].astype(int)

    return np.argsort(-np.abs(beta))[:k].astype(int)


class Solver:
    """
    Usage:
      S = Solver(models=[...], k_top=10)
      out0 = S.fit(X, y, run_id=0, seed=1)
      out1 = S.fit(X, y, run_id=1, seed=2)
    """

    def __init__(self, models, k_top: int = 10):
        self.models = list(models)
        self.k_top = int(k_top)
        self.best_params: Dict[str, Dict[str, Any]] = {}

        # Default grids (you can override per-method via overrides)
        gamma_grid_mcp = (1.5, 3.0, 5.0, 10.0)
        gamma_grid_scad = (2.5, 3.7, 5.0, 10.0)
        alpha_grid_default = (1.0, 0.9, 0.5, 0.1)

        self.registry: Dict[str, MethodAPI] = {
            # sklearn baselines we can keep these, or replace with NCV_Lasso/NCV_ElasticNet below
            "Lasso": MethodAPI(
                tune_fn=tune_lasso_once,
                solve_fn=solve_lasso_fixed,
                default_tune_kwargs={"cv": 5},
                default_solve_kwargs={},
            ),
            "ElasticNet": MethodAPI(
                tune_fn=tune_elastic_net_once,
                solve_fn=solve_elastic_net_fixed,
                default_tune_kwargs={"cv": 5, "l1_ratio_grid": (0.1, 0.5, 0.9, 0.95, 1.0)},
                default_solve_kwargs={},
            ),

            # NEW: ncvreg versions of Lasso / ElasticNet (optional, but you asked for it)
            # penalty="lasso"; gamma irrelevant; alpha controls ridge mixing (alpha=1 => lasso)
            "NCV_Lasso": MethodAPI(
                tune_fn=lambda X, y, **kw: tune_ncvreg_once(
                    X, y, penalty="lasso",
                    alpha_grid=(1.0,),  # pure lasso
                    gamma_grid=None,    # ignored
                    **kw
                ),
                solve_fn=lambda X, y, **kw: solve_ncvreg_fixed(
                    X, y, penalty="lasso",
                    **kw
                ),
                default_tune_kwargs={"nfolds": 5},
                default_solve_kwargs={},
            ),
            "NCV_ElasticNet": MethodAPI(
                tune_fn=lambda X, y, **kw: tune_ncvreg_once(
                    X, y, penalty="lasso",
                    alpha_grid=alpha_grid_default,  # includes <1 => ridge mixing
                    gamma_grid=None,
                    **kw
                ),
                solve_fn=lambda X, y, **kw: solve_ncvreg_fixed(
                    X, y, penalty="lasso",
                    **kw
                ),
                default_tune_kwargs={"nfolds": 5},
                default_solve_kwargs={},
            ),

            # # Precision Lasso (python)
            # "PrecisionLasso": MethodAPI(
            #     tune_fn=tune_precision_lasso_once,
            #     solve_fn=solve_precision_lasso_fixed,
            #     default_tune_kwargs={"cv": 5},
            #     default_solve_kwargs={},
            # ),

            # Precision Lasso (python) - tune lambda by snum-bisection (selection-aligned)
            "PrecisionLasso": MethodAPI(
                tune_fn=tune_precision_lasso_once,
                solve_fn=solve_precision_lasso_fixed,
                # Tune once: target about 2*k nonzeros so top-k ranking is meaningful
                default_tune_kwargs={
                    "snum": 2 * self.k_top,
                    "lr": 1e-3,
                    "max_iter": 500,
                    "patience": 40,
                    "min_lambda": 1e-12,
                    "max_lambda": 1e3,
                    "eps": 1e-6,
                    "logistic": False,
                    # gamma left as None => computed using calculateGamma(X)
                    "gamma": None,
                },
                default_solve_kwargs={},
            ),

            # WLasso (R)
            "WLasso": MethodAPI(
                tune_fn=tune_wlasso_once,
                solve_fn=solve_wlasso_fixed,
                default_tune_kwargs={"cv": 5, "gamma_grid": (0.6, 0.65,0.7, 0.75, 0.8, 0.85, 0.9, 0.95)},
                default_solve_kwargs={"maxsteps": 2000},
            ),

            # MCP / SCAD via ncvreg (R): tune over gamma x alpha, cv inside chooses lambda
            "MCP": MethodAPI(
                tune_fn=lambda X, y, **kw: tune_ncvreg_once(
                    X, y, penalty="MCP",
                    gamma_grid=gamma_grid_mcp,
                    alpha_grid=alpha_grid_default,
                    **kw
                ),
                solve_fn=lambda X, y, **kw: solve_ncvreg_fixed(
                    X, y, penalty="MCP",
                    **kw
                ),
                default_tune_kwargs={"nfolds": 5},
                default_solve_kwargs={},
            ),
            "SCAD": MethodAPI(
                tune_fn=lambda X, y, **kw: tune_ncvreg_once(
                    X, y, penalty="SCAD",
                    gamma_grid=gamma_grid_scad,
                    alpha_grid=alpha_grid_default,
                    **kw
                ),
                solve_fn=lambda X, y, **kw: solve_ncvreg_fixed(
                    X, y, penalty="SCAD",
                    **kw
                ),
                default_tune_kwargs={"nfolds": 5},
                default_solve_kwargs={},
            ),

            # HOLP (R screening)
            "HOLP": MethodAPI(
                tune_fn=tune_holp_once,
                solve_fn=solve_holp_fixed,
                default_tune_kwargs={"cv": 5, "num_select_grid": None},
                default_solve_kwargs={},
            ),

            "our_method": MethodAPI(
                tune_fn=tune_our_method_once,
                solve_fn=solve_our_method_fixed,
                default_tune_kwargs={
                    "cv": 5,
                    "mu_grid": np.logspace(-1, 1.5, 20),#np.arange(0.2, 3.0, 0.2),
                    "pho_grid": np.logspace(-1, 1.0, 20),#(0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                    "max_iter": 1000,
                    "k_select": self.k_top,
                },
                default_solve_kwargs={"max_iter": 1000, "k_select": self.k_top},
            ),
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        run_id: int,
        seed: int = 1,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        k_top: Optional[int] = None,
    ):
        overrides = overrides or {}
        out = {}
        k = int(self.k_top if k_top is None else k_top)

        for name in self.models:
            api = self.registry[name]

            # Tune ONCE on run 0 (only if not already tuned)
            if run_id == 0 and name not in self.best_params:
                tune_kwargs = dict(api.default_tune_kwargs)
                tune_kwargs.update(overrides.get(name, {}))
                best = api.tune_fn(X, y, seed=seed, **tune_kwargs)
                self.best_params[name] = best

            # Solve using cached params (and overrides)
            solve_kwargs = dict(api.default_solve_kwargs)
            solve_kwargs.update(self.best_params.get(name, {}))
            solve_kwargs.update(overrides.get(name, {}))
            res = api.solve_fn(X, y, seed=seed, **solve_kwargs)

            # Standardize top-k selection for benchmarking
            if name == "HOLP":
                res["selected_topk"] = np.asarray(res["selected"], dtype=int)[:k]
            else:
                res["selected_topk"] = _topk_from_beta(res["beta"], k, X=X, y=y)

            out[name] = res

        return out