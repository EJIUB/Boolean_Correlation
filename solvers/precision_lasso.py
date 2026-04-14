import time
import numpy as np
import os, sys

BASE = os.path.dirname(__file__)
PL_ROOT = os.path.abspath(os.path.join(BASE, "..", "src", "thePrecisionLasso"))

sys.path.append(PL_ROOT)
from src.thePrecisionLasso.models.PrecisionLasso import PrecisionLasso


def _safe_calculate_gamma(pl_obj, X, fallback_gamma: float = 0.5):
    """
    Robust gamma calculation for thePrecisionLasso.

    Some versions of calculateGamma() subsample with replace=False and can crash
    when the requested sample size exceeds the population.
    We try:
      1) default calculateGamma(X)
      2) calculateGamma(X, sample=False) if supported
      3) fallback constant gamma

    Returns
    -------
    gamma_used : float
    used_fallback : bool
    """
    # 1) default behavior
    try:
        pl_obj.calculateGamma(X)
        return float(pl_obj.gamma), False
    except Exception:
        pass

    # 2) try disabling sampling if supported
    try:
        pl_obj.calculateGamma(X, sample=False)
        return float(pl_obj.gamma), False
    except TypeError:
        # signature may not accept keyword args in some versions
        pass
    except Exception:
        pass

    # 3) last resort: safe constant
    return float(fallback_gamma), True


def _fit_precision_lasso(
    X, y, lmbd,
    lr=1e-3,
    gamma=None,
    max_iter=200,
    eps=1e-6,
    logistic=False,
    fallback_gamma: float = 0.5,
):
    """
    Fit PrecisionLasso once with fixed hyperparameters.

    Returns
    -------
    beta : (p,) ndarray
    selected : (s,) ndarray (0-based indices where beta != 0)
    meta : dict
    """

    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    model = PrecisionLasso(
        lmbd=float(lmbd),
        eps=float(eps),
        maxIter=int(max_iter),
        lr=float(lr),
        logistic=bool(logistic),
    )
    model.setLogisticFlag(bool(logistic))
    model.setLambda(float(lmbd))
    model.setLearningRate(float(lr))

    # gamma: either user provided or computed robustly
    gamma_fallback = False
    if gamma is None:
        gamma_used, gamma_fallback = _safe_calculate_gamma(model, X, fallback_gamma=fallback_gamma)
        model.setGamma(float(gamma_used))
    else:
        model.setGamma(float(gamma))
        gamma_used = float(gamma)

    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0

    beta = np.asarray(model.getBeta(), dtype=float)  # excludes intercept
    selected = np.flatnonzero(beta != 0).astype(int)

    meta = {
        "runtime_sec": float(dt),
        "lambda": float(lmbd),
        "lr": float(lr),
        "gamma": float(gamma_used),
        "gamma_fallback": bool(gamma_fallback),
        "max_iter": int(max_iter),
        "eps": float(eps),
        "logistic": bool(logistic),
    }
    return beta, selected, meta


def tune_precision_lasso_once(
    X, y,
    seed=1,
    snum=10,                 # target number of non-zeros (use k or 2k)
    lr=1e-3,
    gamma=None,
    max_iter=200,
    eps=1e-6,
    logistic=False,
    patience=40,
    min_lambda=1e-12,
    max_lambda=1e+3,
    fallback_gamma: float = 0.5,
):
    """
    Tune lambda to get ~snum nonzeros using log-space bisection.
    Gamma is computed once robustly (or fixed if provided).

    Returns a dict of parameters that will be merged into solve kwargs.
    """
    np.random.default_rng(seed)  # keep seed usage consistent

    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    # Compute gamma once on full X if not provided (robust)
    gamma_used = gamma
    gamma_fallback = False
    if gamma_used is None:
        #from models.PrecisionLasso import PrecisionLasso
        tmp = PrecisionLasso()
        gamma_used, gamma_fallback = _safe_calculate_gamma(tmp, X, fallback_gamma=fallback_gamma)

    lo = float(min_lambda)
    hi = float(max_lambda)

    best_lam = float(np.exp((np.log(lo) + np.log(hi)) / 2.0))

    for _ in range(int(patience)):
        lam = float(np.exp((np.log(lo) + np.log(hi)) / 2.0))

        beta, sel, _meta = _fit_precision_lasso(
            X, y,
            lmbd=lam,
            lr=lr,
            gamma=gamma_used,           # keep gamma fixed during tuning
            max_iter=max_iter,
            eps=eps,
            logistic=logistic,
            fallback_gamma=fallback_gamma,
        )
        c = int(np.atleast_1d(sel).size)
        best_lam = lam

        if c < snum:
            # too sparse -> reduce lambda
            hi = lam
        elif c > snum:
            # too dense -> increase lambda
            lo = lam
        else:
            break

    # IMPORTANT: These keys may be merged into solve kwargs.
    # It's OK to keep snum/gamma_fallback here as long as solve accepts extra kwargs.
    return {
        "lmbd": float(best_lam),
        "lr": float(lr),
        "gamma": float(gamma_used),
        "gamma_fallback": bool(gamma_fallback),
        "max_iter": int(max_iter),
        "eps": float(eps),
        "logistic": bool(logistic),
        "snum": int(snum),
        "fallback_gamma": float(fallback_gamma),
    }


def solve_precision_lasso_fixed(
    X, y,
    lmbd,
    lr=1e-3,
    gamma=None,
    max_iter=200,
    eps=1e-6,
    logistic=False,
    seed=1,
    fallback_gamma: float = 0.5,
    gamma_fallback=None,  # accept/ignore (tuner metadata)
    snum=None,            # accept/ignore (tuner metadata)
    **kwargs              # accept/ignore any future tuner fields
):
    """
    Solve PrecisionLasso with fixed hyperparameters.
    Accepts extra kwargs so Solver can safely pass cached tuning metadata.
    """
    beta, selected, meta = _fit_precision_lasso(
        X, y,
        lmbd=lmbd,
        lr=lr,
        gamma=gamma,
        max_iter=max_iter,
        eps=eps,
        logistic=logistic,
        fallback_gamma=fallback_gamma,
    )
    return {"beta": beta, "selected": selected, "meta": meta}