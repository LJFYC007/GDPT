import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from LinearSolver import LinearSolver


class GeneralizedPoissonReconstructor:
    def __init__(self, eps=1e-6, verbose=False):
        self.eps = eps
        self.solver = LinearSolver(verbose)

    def buildMatrixS(self, H, W, N, M):
        # S matrix relates image pixels to measurements [I, Dx, Dy]
        # Each row represents one pixel, columns are [pixel_values | x_gradients | y_gradients]
        rows, cols, vals = [], [], []
        offsetX = N                    # start of x-gradient block
        offsetY = N + H * (W - 1)      # start of y-gradient block

        for y in range(H):
            for x in range(W):
                i = y * W + x
                rows.append(i); cols.append(i); vals.append(1.0)  # pixel value

                # x-gradient connections
                if x < W - 1:             # (x,y) -> (x+1, y)
                    j = offsetX + y * (W - 1) + x
                    rows.append(i); cols.append(j); vals.append(-1.0)
                if x > 0:                 # (x,y) -> (x-1, y)
                    j = offsetX + y * (W - 1) + (x - 1)
                    rows.append(i); cols.append(j); vals.append(+1.0)

                # y-gradient connections
                if y < H - 1:             # (x,y) -> (x, y+1)
                    j = offsetY + y * W + x
                    rows.append(i); cols.append(j); vals.append(-1.0)
                if y > 0:                 # (x,y) -> (x, y-1)
                    j = offsetY + (y - 1) * W + x
                    rows.append(i); cols.append(j); vals.append(+1.0)

        return sp.csr_matrix((vals, (rows, cols)), shape=(N, M))

    def buildVarianceWeights(self, H, W, varI, varX, varY):
        N = H * W
        M = N + H * (W - 1) + W * (H - 1)

        def cleanVariance(v):
            v = v.astype(np.float64, copy=False)
            bad = ~np.isfinite(v)
            if np.any(bad):
                v[bad] = np.median(v[~bad])
            return np.maximum(v, 0.0)

        varIFlat = cleanVariance(varI.reshape(-1))
        varXFlat = cleanVariance(varX[:H, :W-1].reshape(-1))
        varYFlat = cleanVariance(varY[:H-1, :W].reshape(-1))

        allVar = np.concatenate([varIFlat, varXFlat, varYFlat])
        return 1.0 / (allVar + self.eps)  # F^{-1} diagonal

    def reconstructChannel(self, img, variance, gradX, gradY, varX, varY, channel=None):
        H, W = img.shape
        N = H * W
        M = 3 * N - W - H

        S = self.buildMatrixS(H, W, N, M)
        weights = self.buildVarianceWeights(H, W, variance, varX, varY)

        # Measurement vector c = [I, Dx, Dy] (same order as S matrix columns)
        imgFlat = img.reshape(-1)
        gxFlat = np.asarray(gradX, dtype=np.float64)[:H, :W-1].reshape(-1)
        gyFlat = np.asarray(gradY, dtype=np.float64)[:H-1, :W].reshape(-1)
        measurements = np.concatenate([imgFlat, gxFlat, gyFlat])

        # Generalized Poisson: A*x = b where A = S*F^{-1}*S^T, b = S*F^{-1}*c
        # x = unknown reconstructed image pixels (what we solve for)
        # A = weighted system matrix combining gradients and pixel constraints
        # b = weighted measurements incorporating all available data
        b = S @ (weights * measurements)

        def matVec(v):
            return S @ (weights * (S.T @ v))  # A*v = S*F^{-1}*S^T*v

        # Diagonal preconditioner: diag(A) = diag(S*F^{-1}*S^T)
        Sd2 = S.power(2)
        diagA = np.maximum(Sd2 @ weights, 1e-12)
        precond = spla.LinearOperator((N, N), matvec=lambda v: v / diagA)

        A = spla.LinearOperator(shape=(N, N), matvec=matVec, dtype=np.float64)
        x, info = self.solver.solveCG(A, b, img.ravel(), channel, precond)
        return x.reshape(H, W)

    def reconstruct(self, img, variance, gradX, gradY, varX, varY):
        H, W, C = img.shape
        result = np.zeros_like(img, dtype=np.float32)

        for ch in range(C):
            result[..., ch] = self.reconstructChannel(
                img[..., ch], variance[..., ch],
                gradX[..., ch], gradY[..., ch],
                varX[..., ch], varY[..., ch], ch)

        return result
