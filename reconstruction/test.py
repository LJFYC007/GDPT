#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


EPSILON_DEFAULT = 1e-4
TOLERANCE_DEFAULT = 1e-6
ITERATIONS_DEFAULT = 10
MIN_DIVISOR = 1e-12


def read_exr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read EXR file: {path}")
    return img.astype(np.float32, copy=False)


def write_exr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image.astype(np.float32))


def load_inv_variance(variance: np.ndarray, epsilon: float) -> np.ndarray:
    cleaned = np.where(np.isfinite(variance), variance, 0.0).astype(np.float32, copy=False)
    np.maximum(cleaned, 0.0, out=cleaned)
    return 1.0 / (cleaned + np.float32(epsilon))


def apply_weighted_system(values: np.ndarray,
                          inv_var_i: np.ndarray,
                          inv_var_x: np.ndarray,
                          inv_var_y: np.ndarray) -> np.ndarray:
    accum = inv_var_i * values

    grad_x = values[:, 1:, :] - values[:, :-1, :]
    weighted_x = inv_var_x[:, :-1, :] * grad_x
    accum[:, :-1, :] -= weighted_x
    accum[:, 1:, :] += weighted_x

    grad_y = values[1:, :, :] - values[:-1, :, :]
    weighted_y = inv_var_y[:-1, :, :] * grad_y
    accum[:-1, :, :] -= weighted_y
    accum[1:, :, :] += weighted_y

    return accum


def compute_b(base: np.ndarray,
              grad_x: np.ndarray,
              grad_y: np.ndarray,
              inv_var_i: np.ndarray,
              inv_var_x: np.ndarray,
              inv_var_y: np.ndarray) -> np.ndarray:
    result = inv_var_i * base

    edge_x = grad_x[:, :-1, :]
    weight_x = inv_var_x[:, :-1, :]
    result[:, :-1, :] -= weight_x * edge_x
    result[:, 1:, :] += weight_x * edge_x

    edge_y = grad_y[:-1, :, :]
    weight_y = inv_var_y[:-1, :, :]
    result[:-1, :, :] -= weight_y * edge_y
    result[1:, :, :] += weight_y * edge_y

    return result


def conjugate_gradient(base: np.ndarray,
                       variance: np.ndarray,
                       grad_x: np.ndarray,
                       grad_y: np.ndarray,
                       var_x: np.ndarray,
                       var_y: np.ndarray,
                       iterations: int,
                       epsilon: float,
                       tolerance: float):
    base_rgb = base[..., :3]

    inv_var_i = load_inv_variance(variance[..., :3], epsilon)
    inv_var_x = load_inv_variance(var_x[..., :3], epsilon)
    inv_var_y = load_inv_variance(var_y[..., :3], epsilon)

    b = compute_b(base_rgb, grad_x[..., :3], grad_y[..., :3], inv_var_i, inv_var_x, inv_var_y)

    x = base_rgb.copy()
    Ax = apply_weighted_system(base_rgb, inv_var_i, inv_var_x, inv_var_y)
    r = b - Ax
    p = r.copy()

    residual_norm = float(np.sum(r * r))
    print(f"Initial residual norm = {np.sqrt(residual_norm):.6e}")

    if not np.isfinite(residual_norm) or residual_norm <= tolerance:
        return x, residual_norm, []

    history = []
    prev_residual = residual_norm

    for iteration in range(iterations):
        prev_residual_old = prev_residual
        Ap = apply_weighted_system(p, inv_var_i, inv_var_x, inv_var_y)
        dot_p_ap = float(np.sum(p * Ap))

        if not np.isfinite(dot_p_ap) or abs(dot_p_ap) < MIN_DIVISOR:
            print(f"Iteration {iteration}: dot(p, Ap) = {dot_p_ap:.6e}, stopping due to degeneracy.")
            break

        alpha = prev_residual_old / dot_p_ap
        x = x + alpha * p
        r = r - alpha * Ap

        new_residual = float(np.sum(r * r))
        history.append({
            "iteration": iteration,
            "prev_residual": prev_residual_old,
            "new_residual": new_residual,
            "alpha": alpha,
            "beta": None,
        })

        if not np.isfinite(new_residual) or new_residual <= tolerance:
            print(f"Iteration {iteration}: residual norm = {np.sqrt(new_residual):.6e}, stopping by tolerance.")
            prev_residual = new_residual
            break

        beta = new_residual / prev_residual_old
        p = r + beta * p

        history[-1]["beta"] = beta

        print(
            f"Iteration {iteration}: "
            f"prev residual = {np.sqrt(prev_residual_old):.6e}, "
            f"new residual = {np.sqrt(new_residual):.6e}, "
            f"alpha = {alpha:.6e}, beta = {beta:.6e}"
        )

        prev_residual = new_residual

    final_residual = float(np.sum(r * r))
    return x, final_residual, history


def load_inputs(output_dir: Path, spp: int):
    def require(name: str) -> np.ndarray:
        path = output_dir / name
        return read_exr(path)

    base = require(f"Mogwai.AccumulatePass.output.{spp}.exr")
    variance = require(f"Mogwai.PostProcess.Output.{spp}.exr")
    grad_x = require(f"Mogwai.ErrorMeasureXPass.Output.{spp}.exr")
    grad_y = require(f"Mogwai.ErrorMeasureYPass.Output.{spp}.exr")
    var_x = require(f"Mogwai.PostProcessX.Output.{spp}.exr")
    var_y = require(f"Mogwai.PostProcessY.Output.{spp}.exr")

    return base, variance, grad_x, grad_y, var_x, var_y


def compare_with_gpu(solution_rgb: np.ndarray, gpu_path: Path) -> None:
    if not gpu_path.exists():
        print("Compute shader output not found; skipping comparison.")
        return

    gpu_image = read_exr(gpu_path)[..., :3]
    diff = solution_rgb - gpu_image
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    print(f"GPU comparison: MSE = {mse:.6e}, max |diff| = {max_abs:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Replicate ReconstructionPass compute shader in Python.")
    default_output_dir = (Path(__file__).resolve().parents[1] / "output").resolve()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory containing Mogwai outputs.")
    parser.add_argument("--spp", type=int, default=16, help="Samples per pixel to load.")
    parser.add_argument("--iterations", type=int, default=ITERATIONS_DEFAULT, help="Maximum CG iterations.")
    parser.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT, help="Variance regularization term.")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_DEFAULT, help="Residual tolerance for early exit.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save reconstructed EXR.")
    parser.add_argument("--compare-gpu", action="store_true", help="Compare with compute shader output if available.")
    args = parser.parse_args()

    base, variance, grad_x, grad_y, var_x, var_y = load_inputs(args.output_dir, args.spp)

    solution_rgb, final_residual, history = conjugate_gradient(
        base,
        variance,
        grad_x,
        grad_y,
        var_x,
        var_y,
        iterations=args.iterations,
        epsilon=args.epsilon,
        tolerance=args.tolerance,
    )

    print(f"Final residual norm = {np.sqrt(final_residual):.6e}")

    if args.compare_gpu:
        gpu_path = args.output_dir / f"Mogwai.ReconstructionPass.output.{args.spp}.exr"
        compare_with_gpu(solution_rgb, gpu_path)

    if args.save is not None:
        alpha = base[..., 3:4] if base.shape[2] > 3 else np.ones_like(base[..., :1])
        output = np.concatenate((solution_rgb, alpha), axis=2)
        write_exr(args.save, output)
        print(f"Saved reconstructed image to {args.save}")


if __name__ == "__main__":
    main()
