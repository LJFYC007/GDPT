import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

result_dir = "../minimal_result"
reference_file = "reference-staircase.exr"
methods = ["pt", "simple", "poisson"]
spp_values = [32, 64, 128, 1024, 50000]

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

    # Store all MSE results in a nested dictionary
    mse_results = {}

    for method in methods:
        mse_results[method] = {}
        for spp in spp_values:
            input_file = f"{method}-{spp}.exr"
            input_path = os.path.join(result_dir, input_file)
            input_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            if input_img is None:
                # Don't show warning, just skip this file
                mse_results[method][spp] = None
                continue

            try:
                difference_output = os.path.join(result_dir, f"difference-{method}-{spp}spp.exr")
                mse = calculate_mse_and_save_difference(input_img, reference_img, difference_output)
                mse_results[method][spp] = mse
            except Exception as e:
                mse_results[method][spp] = None

    # Display results in table format
    print_mse_table(mse_results)
    return mse_results

def print_mse_table(mse_results):
    print("\n=== MSE Comparison Results (Table Format) ===")

    # Print header
    header = "Method".ljust(12)
    for spp in spp_values:
        header += f"{spp}spp".ljust(15)
    print(header)
    print("-" * len(header))

    # Print each method's results
    for method in methods:
        row = method.ljust(12)
        for spp in spp_values:
            mse = mse_results.get(method, {}).get(spp)
            if mse is not None:
                row += f"{mse:.6f}".ljust(15)
            else:
                row += " ".ljust(15)  # Empty space for missing files
        print(row)

    print()  # Empty line after table

if __name__ == "__main__":
    results = compare_images_with_reference()
