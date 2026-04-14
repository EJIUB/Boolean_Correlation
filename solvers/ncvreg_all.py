import json, subprocess, time, tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def _as_scalar(x):
    """Convert JSON value to a scalar float if possible; otherwise return None."""
    if x is None:
        return None
    if isinstance(x, dict):
        # try values inside dict
        for v in x.values():
            try:
                return float(v)
            except Exception:
                continue
        return None
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        try:
            return float(x[0])
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None
    


def _call_ncvreg(X, y, *, penalty, nfolds, seed,
                gamma_grid=None, alpha_grid=None,
                lambda_fixed=None, gamma_fixed=None, alpha_fixed=None,
                rscript="Rscript"):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x_csv, y_csv, out_json = td/"X.csv", td/"y.csv", td/"out.json"
        pd.DataFrame(X).to_csv(x_csv, index=False, header=False)
        pd.DataFrame(y).to_csv(y_csv, index=False, header=False)

        gamma_grid_str = "" if gamma_grid is None else ",".join(map(str, gamma_grid))
        alpha_grid_str = "" if alpha_grid is None else ",".join(map(str, alpha_grid))

        cmd = [
            rscript, "src/ncvreg/run_ncvreg.R",
            str(x_csv), str(y_csv), str(out_json),
            str(penalty), str(int(nfolds)), str(int(seed)),
            gamma_grid_str, alpha_grid_str,
        ]

        # fixed mode arguments
        if lambda_fixed is not None:
            cmd += [str(float(lambda_fixed)),
                    str(float(gamma_fixed)) if gamma_fixed is not None else "NA",
                    str(float(alpha_fixed)) if alpha_fixed is not None else "NA"]

        t0 = time.perf_counter()
        #subprocess.check_call(cmd)
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        #print("STDOUT:\n", proc.stdout)
        #print("STDERR:\n", proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError("R script failed")
        
        dt = time.perf_counter() - t0

        res = json.loads(out_json.read_text())
        
        beta = np.array(res["beta_hat"], float)
        selected = np.array(res.get("selected_1based", []), int) - 1

        return {
            "beta": beta,
            "selected": selected,
            "meta": {
                "runtime_sec": dt,
                "penalty": res.get("penalty"),
                "alpha": res.get("alpha"),
                "gamma": res.get("gamma"),
                "lambda": res.get("lambda"),
                "cve_min": res.get("cve_min"),
            }
        }

def tune_ncvreg_once(X, y, penalty, nfolds=5, seed=1,
                     gamma_grid=None, alpha_grid=None):
    out = _call_ncvreg(X, y, penalty=penalty, nfolds=nfolds, seed=seed,
                      gamma_grid=gamma_grid, alpha_grid=alpha_grid,
                      lambda_fixed=None)
    # what to cache for runs 1..9
    return {
        "lambda_fixed": float(out["meta"]["lambda"]),
        "alpha_fixed": float(out["meta"]["alpha"]),
        # "gamma_fixed": None if out["meta"]["gamma"] is None else float(out["meta"]["gamma"]),
        "gamma_fixed": _as_scalar(out["meta"]["gamma"]),
        "nfolds": int(nfolds),
    }

def solve_ncvreg_fixed(X, y, penalty, lambda_fixed, alpha_fixed=1.0, gamma_fixed=None, nfolds=5, seed=1):
    return _call_ncvreg(X, y, penalty=penalty, nfolds=nfolds, seed=seed,
                       lambda_fixed=lambda_fixed, alpha_fixed=alpha_fixed, gamma_fixed=gamma_fixed)