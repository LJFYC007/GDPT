import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def construct_S(H, W, N, M):
    rows, cols, vals = [], [], []
    off_x = N                   # Dx block offset
    off_y = N + H * (W - 1)     # Dy block offset

    def pixel_index(x, y):
        return y * W + x

    for y in range(H):
        for x in range(W):
            i = pixel_index(x, y)

            # 点值块
            rows.append(i); cols.append(i); vals.append(1.0)

            # 水平梯度块
            if x < W - 1:  # 右边边 (x,y)->(x+1,y)
                j = off_x + y * (W - 1) + x
                rows.append(i); cols.append(j); vals.append(-1.0)
            if x > 0:      # 左边边 (x-1,y)->(x,y)
                j = off_x + y * (W - 1) + (x - 1)
                rows.append(i); cols.append(j); vals.append(+1.0)

            # 垂直梯度块
            if y < H - 1:  # 下边边 (x,y)->(x,y+1)
                j = off_y + y * W + x
                rows.append(i); cols.append(j); vals.append(-1.0)
            if y > 0:      # 上边边 (x,y-1)->(x,y)
                j = off_y + (y - 1) * W + x
                rows.append(i); cols.append(j); vals.append(+1.0)

    S = sp.csr_matrix((vals, (rows, cols)), shape=(N, M))
    return S

def construct_F(H, W, variance_I, variance_dx, variance_dy, eps=1e-6):
    N = H * W
    M = N + H * (W - 1) + W * (H - 1)

    # 展平顺序与 S 的列顺序一致：
    # [ I_flat | Dx_row_major_flat | Dy_row_major_flat ]
    var_I_flat  = variance_I.reshape(-1)                         # 长度 N
    var_Dx_flat = variance_dx[:H, :W-1].reshape(-1)                        # 长度 H*(W-1)
    var_Dy_flat = variance_dy[:H-1, :W].reshape(-1)                        # 长度 (H-1)*W

    # 清理异常值：NaN/Inf -> 用大的方差替代（小权重）
    def sanitize(v):
        v = v.astype(np.float64, copy=False)
        bad = ~np.isfinite(v)
        if np.any(bad):
            # 用非坏值的中位数
            med = np.median(v[~bad])
            v[bad] = med
        v = np.maximum(v, 0.0)
        return v

    var_I_flat  = sanitize(var_I_flat)
    var_Dx_flat = sanitize(var_Dx_flat)
    var_Dy_flat = sanitize(var_Dy_flat)

    diag_var = np.concatenate([var_I_flat, var_Dx_flat, var_Dy_flat], axis=0)
    assert diag_var.shape[0] == M
    diag_vec = 1.0 / (diag_var + eps)
    return diag_vec

def generalized_poisson_reconstruct(I0, variance, grad_x, grad_y, variance_x, variance_y):
    H, W = I0.shape
    N = H * W
    M = 3 * N - W - H
    S = construct_S(H, W, N, M)
    F_inv_diag = construct_F(H, W, variance, variance_x, variance_y)

    # --- 构造测量向量 c = [I, Dx, Dy] (与 S/F 的列顺序严格一致) ---
    I_flat = I0.reshape(-1)
    gx = np.asarray(grad_x, dtype=np.float64)[:H, :W-1].reshape(-1)
    gy = np.asarray(grad_y, dtype=np.float64)[:H-1, :W].reshape(-1)
    c = np.concatenate([I_flat, gx, gy], axis=0)
    assert c.size == M

    # --- 计算 b = S F^{-1} c 与 A = S F^{-1} S^T ---
    b = S @ (F_inv_diag * c)

    # A = S @ sp.diags(F_inv_diag, 0, shape=(M, M), format='csr') @ S.T
    # x = spla.spsolve(A.tocsc(), b)   # 对称正定时表现良好
    # I_hat = x.reshape(H, W)
    # return I_hat

    # # 定义 A 的乘法：A*v = S * (F^{-1} * (S^T * v))
    def matvec(v):
        tmp = S.T @ v                # (M)
        tmp = F_inv_diag * tmp       # (M)
        Av  = S @ tmp                # (N)
        return Av

    res_history = []
    def cb_xk(xk):
        r = b - A_linop @ xk
        res = np.linalg.norm(r)
        res_history.append(res)
        print(f"iter {len(res_history):3d}: residual = {res:.6e}")

    # diag(A) = (S.^2) @ F_inv_diag    （因为 S 元素只在 {0,±1}，平方成 0/1）
    Sd2   = S.power(2)                        # 稀疏0/1
    diagA = Sd2 @ F_inv_diag                  # 形状 (N,)
    # 保底避免 0
    diagA = np.maximum(diagA, 1e-12)
    Minv  = 1.0 / diagA

    def precond(v):
        return Minv * v

    A_linop = spla.LinearOperator(shape=(N, N), matvec=matvec, dtype=np.float64)
    M = spla.LinearOperator((N,N), matvec=precond, dtype=np.float64)
    x, info = spla.cg(A_linop, b, M=M, rtol=1e-8, atol=0.0, maxiter=300, callback=cb_xk)
    return x.reshape(H, W)

# Process different SPP values for gradients
spp_values = [50000]
# spp_values = [32, 64, 128, 1024, 50000]
for spp in spp_values:
    pt = cv2.imread(f"../output/Mogwai.AccumulatePass.output.{spp}.exr", cv2.IMREAD_UNCHANGED)
    gradX = cv2.imread(f"../output/Mogwai.ErrorMeasureXPass.Output.{spp}.exr", cv2.IMREAD_UNCHANGED)
    gradY = cv2.imread(f"../output/Mogwai.ErrorMeasureYPass.Output.{spp}.exr", cv2.IMREAD_UNCHANGED)
    variance = cv2.imread(f"../output/Mogwai.AccumulatePass.variance.{spp}.exr", cv2.IMREAD_UNCHANGED)
    varianceX = cv2.imread(f"../output/Mogwai.AccumulatePassX.variance.{spp}.exr", cv2.IMREAD_UNCHANGED)
    varianceY = cv2.imread(f"../output/Mogwai.AccumulatePassY.variance.{spp}.exr", cv2.IMREAD_UNCHANGED)

    H, W, C = pt.shape
    reconstructed = np.zeros_like(pt, dtype=np.float32)
    for channel in range(C):
        reconstructed[..., channel] = generalized_poisson_reconstruct(pt[..., channel], variance[..., channel], gradX[..., channel], gradY[..., channel], varianceX[..., channel], varianceY[..., channel])
    cv2.imwrite(f"../minimal_result/generalized-poisson-{spp}.exr", reconstructed.astype(np.float32))
    print(f"Completed reconstruction with gradient {spp}spp and pt {spp}spp")
