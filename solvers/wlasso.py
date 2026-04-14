import json, subprocess, time, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def _call_wlasso_R(X, y, gamma, maxsteps=2000, rscript="Rscript"):
    X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x_csv, y_csv, out_json = td/"X.csv", td/"y.csv", td/"out.json"
        pd.DataFrame(X).to_csv(x_csv, index=False, header=False)
        pd.DataFrame(y).to_csv(y_csv, index=False, header=False)
        cmd = [rscript, "src/wlasso/run_wlasso.R", str(x_csv), str(y_csv), str(out_json), str(float(gamma)), str(int(maxsteps))]
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
        return beta, sel, dt

def tune_wlasso_once(X, y, gamma_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), cv=5, seed=10, maxsteps=1000):
    X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1)
    kf = KFold(n_splits=int(cv), shuffle=True, random_state=int(seed))
    best_g, best_score = None, float("inf")
    for g in gamma_grid:
        scores=[]
        for tr, te in kf.split(X):
            beta, _, _ = _call_wlasso_R(X[tr], y[tr], gamma=g, maxsteps=maxsteps)
            scores.append(float(np.mean((y[te] - X[te] @ beta)**2)))
        s = float(np.mean(scores))
        if s < best_score:
            best_score, best_g = s, float(g)
    return {"gamma": best_g, "maxsteps": int(maxsteps)}

def solve_wlasso_fixed(X, y, gamma=0.95, maxsteps=2000, seed=1):
    beta, sel, dt = _call_wlasso_R(X, y, gamma=gamma, maxsteps=maxsteps)
    return {"beta": beta, "selected": sel, "meta": {"runtime_sec": dt, "gamma": float(gamma), "maxsteps": int(maxsteps)}}