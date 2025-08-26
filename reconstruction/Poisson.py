import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg

class PoissonReconstructor:
    def __init__(self, lambd=0.1, verbose=False):
        self.lambd = lambd
        self.verbose = verbose

    def computeDivergence(self, gx, gy):
        div = np.zeros_like(gx)
        div[0, :, :] -= gx[0, :, :]                          # top row
        div[1:-1, :, :] += gx[:-2, :, :] - gx[1:-1, :, :]
        div[-1, :, :] += gx[-2, :, :]                        # bottom row
        div[:, 0, :] -= gy[:, 0, :]                          # left column
        div[:, 1:-1, :] += gy[:, :-2, :] - gy[:, 1:-1, :]
        div[:, -1, :] += gy[:, -2, :]                        # right column
        return div

    def computeLaplacian(self, img):
        lap = -np.zeros_like(img, dtype=np.float32)
        deg = np.zeros_like(img, dtype=np.float32)

        lap[:-1] -= img[1:]
        lap[1:] -= img[:-1]
        deg[:-1] += 1; deg[1:] += 1

        lap[:, :-1] -= img[:, 1:]
        lap[:, 1:] -= img[:, :-1]
        deg[:, :-1] += 1; deg[:, 1:] += 1

        return lap + deg * img

    def reconstructChannel(self, gradX, gradY, initialImg, channel=None):
        H, W = initialImg.shape

        gx3d = gradX[..., np.newaxis]
        gy3d = gradY[..., np.newaxis]
        img3d = initialImg[..., np.newaxis]

        # Poisson equation: (Laplacian + λI) * x = div(grad) + λ * initialImg
        # b = right hand side: divergence of gradients + regularization term
        rhs = self.computeDivergence(gx3d, gy3d) + self.lambd * img3d.astype(np.float32)
        b = rhs[..., 0].ravel()

        # A = left hand side operator: (Laplacian + λI)
        # x = unknown reconstructed image (what we solve for)
        def matVec(v):
            vImg = v.reshape(H, W, 1)
            return (self.computeLaplacian(vImg) + self.lambd * vImg).ravel()

        A = LinearOperator((H*W, H*W), matvec=matVec, dtype=np.float32)

        def callback(xk):
            r = b - A @ xk
            res = np.linalg.norm(r)
            print(f"[ch {channel}] residual = {res:.6e}")
        if self.verbose:
            sol, info = scipy.sparse.linalg.cg(A, b, x0=initialImg.ravel(), rtol=1e-10, maxiter=1000, callback=callback)
        else:
            sol, info = scipy.sparse.linalg.cg(A, b, x0=initialImg.ravel(), rtol=1e-10, maxiter=1000)
        return sol.reshape(H, W)

    def reconstruct(self, gradX, gradY, initialImg):
        H, W, C = initialImg.shape
        result = np.zeros_like(initialImg, dtype=np.float32)

        for ch in range(C):
            result[..., ch] = self.reconstructChannel(
                gradX[..., ch], gradY[..., ch], initialImg[..., ch], ch)

        return result
