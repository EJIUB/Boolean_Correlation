import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import time
import bottleneck as bn
import sys
import os
import random
import gurobipy as gp
from gurobipy import GRB

# import PQN
#sys.path.append(os.path.abspath("./PQN_Python"))
BASE = os.path.dirname(__file__)
PQN_ROOT = os.path.join(BASE, "PQN_Python")

# Add BOTH PQN root AND its submodules
sys.path.append(PQN_ROOT)
#sys.path.append(os.path.join(PQN_ROOT, "minConf"))
#sys.path.append(os.path.join(PQN_ROOT, "minFunc"))
from minConf.minConf_PQN import minConF_PQN


def ProjCappedSimplex_NIPS21_Fix(u, k):

    original_shape = u.shape
    u = np.asarray(u).reshape(-1)   # 只用于内部计算

    if u.sum() <= k and np.all((u >= 0) & (u <= 1)):
        return u.reshape(original_shape)

    r = np.partition(u, -k)[-k] - 0.5

    v = np.empty_like(u)
    tol = 1e-30

    for _ in range(500):
        np.subtract(u, r, out=v)
        np.clip(v, 0.0, 1.0, out=v)

        grad = k - v.sum()
        if abs(grad) < tol:
            break

        active = np.count_nonzero((v > 0.0) & (v < 1.0))
        if active == 0:
            break

        r -= grad / active

    return v.reshape(original_shape)


def ProjCappedSimplex_NIPS21_Fix_LinearSearch(u, k):

    original_shape = u.shape
    u = np.asarray(u).reshape(-1)   # 只用于内部计算
    
    # no need for projection
    if u.sum() <= k and np.all((u >= 0) & (u <= 1)):
        #print(u)
        #print("no projectoin ", u.sum())
        return u.reshape(original_shape)
    
    # when to do clip
    mask_between = (u >= 0) & (u < 1)
    sum_between = u[mask_between].sum()
    num_ones = np.count_nonzero(u >= 1)
    if sum_between <= k - num_ones:
        np.clip(u, 0.0, 1.0, out=u)
        return u.reshape(original_shape)

    r = np.partition(u, -k)[-k] - 0.000001

    v = np.empty_like(u)
    tol = 1e-12

    for _ in range(50):
        np.subtract(u, r, out=v)

        np.clip(v, 0.0, 1.0, out=v)
        Ind = np.nonzero(v)

        grad = k - v.sum()
        if abs(grad) < tol:
            break

        active = np.count_nonzero((v > 0.0) & (v < 1.0))
        if active == 0:
            break

        # linear search    
        step = grad / active
        alpha = 1.0

        while True:
            r_new = r - alpha * step
            v_new = np.clip(u - r_new, 0, 1)
            if abs(k - v_new.sum()) < abs(grad):
                break
            alpha *= 0.5

        r = r_new

    return v.reshape(original_shape)



def ProjCSimplex_Gurobi(u, k):
    u = np.asarray(u).ravel()
    n = u.size

    model = gp.Model()
    model.setParam('OutputFlag', 0)

    x = model.addMVar(n, lb=0.0, ub=1.0)

    Q = np.eye(n)
    f = -2 * u

    model.setObjective(x @ Q @ x + f @ x, GRB.MINIMIZE)
    model.addConstr(x.sum() <= k)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed. Status: {model.status}")

    return x.X[:, None]


def Run_PQN(funObj, funProj, u_init, max_iter=100, verbose=3):
    """
    Generic PQN runner.

    Parameters
    ----------
    funObj : callable
        Objective function handle.
        Must return (f, g) where f is objective value and g is gradient.
    funProj : callable
        Projection function handle.
    u_init : ndarray
        Initial point.
    max_iter : int, optional
        Maximum number of iterations.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    uout : ndarray
        Optimized solution.
    obj : ndarray
        Objective values.
    info : dict
        Additional solver info.
    """

    options = {
        'verbose': verbose,
        'maxIter': max_iter
    }

    uout, obj, info = minConF_PQN(funObj, u_init, funProj, options)

    return uout, obj, info
