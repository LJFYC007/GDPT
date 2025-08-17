import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

def divergence(gx, gy):
    div = np.zeros_like(gx)
    div[0, :, :] -= gx[0, :, :]         # top row
    div[1:-1, :, :] +=  gx[:-2, :, :] - gx[1:-1, :, :]
    div[-1, :, :] += gx[-2, :, :]       # bottom row
    div[:, 0, :] -= gy[:, 0, :]         # left col
    div[:, 1:-1, :] += gy[:, :-2, :] - gy[:, 1:-1, :]
    div[:, -1, :] += gy[:, -2, :]       # right col
    return div

def laplacian_neg(image: np.ndarray) -> np.ndarray:
    lap = -np.zeros_like(image, dtype=np.float32)
    deg = np.zeros_like(image, dtype=np.float32)

    lap[:-1] -= image[1:]
    lap[1:]  -= image[:-1]
    deg[:-1] += 1; deg[1:] += 1

    lap[:, :-1] -= image[:, 1:]
    lap[:, 1:]  -= image[:, :-1]
    deg[:, :-1] += 1; deg[:, 1:] += 1

    lap += deg * image
    return lap

def poisson_reconstruct(grad_x, grad_y, I0):
    lambd = 0.1
    H, W, C = I0.shape
    rhs = divergence(grad_x, grad_y) + lambd * I0.astype(np.float32)
    result = np.zeros_like(I0, dtype=np.float32)

    for ch in range(C):
        b = rhs[..., ch].ravel()
        x0 = I0[..., ch].ravel()

        res_history = []
        def cb_xk(xk):
            r = b - A @ xk
            res = np.linalg.norm(r)
            res_history.append(res)
            print(f"[ch {ch}] iter {len(res_history):3d}: residual = {res:.6e}")

        def mv(v):
            v_img = v.reshape(H, W, 1)
            return (laplacian_neg(v_img) + lambd * v_img).ravel()

        A = LinearOperator((H*W, H*W), matvec=mv, dtype=np.float32)

        sol, info = cg(A, b, x0, rtol=1e-10, atol=0, maxiter=500)
        # sol, info = cg(A, b, x0, rtol=1e-10, atol=0, maxiter=500, callback=cb_xk)

        result[..., ch] = sol.reshape(H, W)

    return result

# it seems grad X Y is swapped, since H, W is swapped

# Process different SPP values for gradients
spp_values = [32, 64, 128, 1024, 50000]
for spp in spp_values:
    pt = cv2.imread(f"../minimal_result/pt-{spp}.exr", cv2.IMREAD_UNCHANGED)
    gradY = cv2.imread(f"../minimal_result/gradientX-{spp}.exr", cv2.IMREAD_UNCHANGED)
    gradX = cv2.imread(f"../minimal_result/gradientY-{spp}.exr", cv2.IMREAD_UNCHANGED)

    reconstructed = poisson_reconstruct(gradX, gradY, pt)
    cv2.imwrite(f"../minimal_result/poisson-{spp}.exr", reconstructed.astype(np.float32))
    print(f"Completed reconstruction with gradient {spp}spp and pt {spp}spp")
