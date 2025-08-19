import numpy as np
from scipy.sparse.linalg import cg


class LinearSolver:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def solveCG(self, A, b, x0=None, channel=None, M=None):
        if self.verbose and channel is not None:
            def callback(xk):
                r = b - A @ xk
                res = np.linalg.norm(r)
                print(f"[ch {channel}] residual = {res:.6e}")
            sol, info = cg(A, b, x0=x0, M=M, rtol=1e-10, maxiter=500, callback=callback)
        else:
            sol, info = cg(A, b, x0=x0, M=M, rtol=1e-10, maxiter=500)
        return sol, info
