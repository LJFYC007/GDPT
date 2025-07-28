import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

result_dir = "../result"
reference_file = "reference-staircase.exr"
methods = ["simple", "pt", "poisson"]

def calculate_mse_and_save_difference(image1, image2, output_path):
    if image1.shape != image2.shape:
        raise ValueError(f"Image shape mismatch: {image1.shape} vs {image2.shape}")
    difference = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    cv2.imwrite(output_path, difference)
    return mse

def compare_images_with_reference():
    reference_path = os.path.join(result_dir, reference_file)
    reference_img = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)
    if reference_img is None:
        raise FileNotFoundError(f"Cannot read reference image: {reference_path}")

    mse_results = {}
    for method in methods:
        input_file = f"{method}-64spp.exr"
        input_path = os.path.join(result_dir, input_file)
        input_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if input_img is None:
            print(f"Warning: Cannot read image {input_path}")
            continue

        difference_output = os.path.join(result_dir, f"difference-{method}-64spp.exr")
        mse = calculate_mse_and_save_difference(input_img, reference_img, difference_output)
        mse_results[method] = mse

    # Output all MSE results
    print("\n=== MSE Comparison Results ===")
    for method, mse in mse_results.items():
        print(f"{method}: {mse:.6f}")

    return mse_results

if __name__ == "__main__":
    results = compare_images_with_reference()
