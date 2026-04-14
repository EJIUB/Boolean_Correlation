import json, subprocess, time, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def _call_ncvreg_R(X, y, penalty, alpha=1.0, nfolds=5, seed=1, lambda_fixed=None, rscript="Rscript"):
    X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x_csv, y_csv, out_json = td/"X.csv", td/"y.csv", td/"out.json"
        pd.DataFrame(X).to_csv(x_csv, index=False, header=False)
        pd.DataFrame(y).to_csv(y_csv, index=False, header=False)
        cmd = [rscript, "src/ncvreg/run_ncvreg.R", str(x_csv), str(y_csv), str(out_json),
               str(penalty), str(float(alpha)), str(int(nfolds)), str(int(seed))]
        if lambda_fixed is not None:
            cmd.append(str(float(lambda_fixed)))
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
        sel = np.array(res.get("selected_1based", []), int) - 1
        # lam = float(res["lambda"])
        lam = float(res.get("lambda", res.get("lambda_min")))
        return beta, sel, lam, dt

def tune_ncvreg_once(X, y, penalty="MCP", alpha=1.0, nfolds=5, seed=1):
    # run CV once, store lambda
    _, _, lam, _ = _call_ncvreg_R(X, y, penalty=penalty, alpha=alpha, nfolds=nfolds, seed=seed, lambda_fixed=None)
    return {"lambda_fixed": lam, "alpha": float(alpha), "nfolds": int(nfolds)}

def solve_ncvreg_fixed(X, y, penalty="MCP", alpha=1.0, nfolds=5, seed=1, lambda_fixed=None):
    beta, sel, lam, dt = _call_ncvreg_R(X, y, penalty=penalty, alpha=alpha, nfolds=nfolds, seed=seed, lambda_fixed=lambda_fixed)
    return {"beta": beta, "selected": sel, "meta": {"runtime_sec": dt, "penalty": penalty, "lambda": lam, "alpha": float(alpha)}}