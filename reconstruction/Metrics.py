import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from skimage.metrics import structural_similarity

class ImageMetrics:
    """Calculate image quality metrics for reconstruction comparison"""

    @staticmethod
    def calculateRSE(img1, img2):
        # Relative Squared Error
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        relativeError = ((img1 - img2) / (img2 + 1e-10)) ** 2
        return np.mean(relativeError)

    @staticmethod
    def calculateRAE(img1, img2):
        # Relative Absolute Error
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        relativeError = np.abs(img1 - img2) / (np.abs(img2) + 1e-10)
        return np.mean(relativeError)

    @staticmethod
    def calculateSSIM(img1, img2):
        # Structural Similarity Index
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Calculate SSIM for each channel and average
        ssimValues = []
        for i in range(img1.shape[2]):
            ssim = structural_similarity(img1[:, :, i], img2[:, :, i], data_range=img2[:, :, i].max() - img2[:, :, i].min())
            ssimValues.append(ssim)
        return np.mean(ssimValues)


class ImageComparator:
    """Compare reconstruction results with reference images"""

    def __init__(self, resultDir, refFile, methods, sppValues, sceneName):
        self.resultDir = resultDir
        self.refFile = refFile
        self.methods = methods
        self.sppValues = sppValues
        self.sceneName = sceneName
        self.metrics = ImageMetrics()

    def compareWithReference(self):
        refPath = self.refFile if os.path.isabs(self.refFile) else os.path.join(self.resultDir, self.refFile)
        refImg = cv2.imread(refPath, cv2.IMREAD_UNCHANGED)
        if refImg is None:
            raise FileNotFoundError(f"Cannot read: {refPath}")

        refImg = refImg[:, :, :3]

        results = {'RSE': {}, 'RAE': {}, 'SSIM': {}}

        for method in self.methods:
            for metric in results.keys():
                results[metric][method] = {}

            for spp in self.sppValues:
                imgPath = os.path.join(self.resultDir, f"{method}", f"{method}_{spp}.exr")
                img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

                if img is not None:
                    img = img[:, :, :3]
                    results['RSE'][method][spp] = self.metrics.calculateRSE(img, refImg)
                    results['RAE'][method][spp] = self.metrics.calculateRAE(img, refImg)
                    results['SSIM'][method][spp] = self.metrics.calculateSSIM(img, refImg)
                else:
                    for metric in results.keys():
                        results[metric][method][spp] = None

        self.printResults(results)
        self.plotResults(results)
        return results

    def printResults(self, results):
        for metric in ['RSE', 'RAE', 'SSIM']:
            print(f"\n=== {self.sceneName} {metric} Results ===")
            header = "Method".ljust(20)
            for spp in self.sppValues:
                header += f"{spp}spp".ljust(15)
            print(header)
            print("-" * len(header))

            for method in self.methods:
                row = method.ljust(20)
                for spp in self.sppValues:
                    value = results[metric].get(method, {}).get(spp)
                    if value is not None:
                        row += f"{value:.6f}".ljust(15)
                    else:
                        row += "N/A".ljust(15)
                print(row)
            print()

    def plotResults(self, results):
        metrics = ['RSE', 'RAE', 'SSIM']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for method_idx, method in enumerate(self.methods):
                # Skip 'pt' method in SSIM plot
                if metric == 'SSIM' and method == 'pt':
                    continue

                spp_list = []
                values = []

                for spp in sorted(self.sppValues):
                    value = results[metric].get(method, {}).get(spp)
                    if value is not None:
                        spp_list.append(spp)
                        values.append(value)

                if spp_list:
                    ax.plot(spp_list, values,
                           marker=markers[method_idx % len(markers)],
                           color=colors[method_idx % len(colors)],
                           linewidth=2, markersize=8,
                           label=method, alpha=0.8)

            ax.set_xlabel('Samples Per Pixel (SPP)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison - {self.sceneName}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', which='both')

            # Set log scale for x-axis
            ax.set_xscale('log', base=2)

            # For SSIM, use linear scale and set appropriate y-axis range
            if metric == 'SSIM':
                # Find min/max SSIM values to set appropriate range (exclude 'pt' method)
                all_values = [v for method in self.methods
                             if method != 'pt'
                             for v in results[metric].get(method, {}).values()
                             if v is not None]
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)
                    # Add some padding and set range
                    y_range = max_val - min_val
                    ax.set_ylim(max(0.80, min_val - y_range * 0.1), min(1.0, max_val + y_range * 0.05))

                    # Custom formatter for SSIM to show 3 decimal places
                    def ssimFormatter(value, pos):
                        return f'{value:.3f}'
                    ax.yaxis.set_major_formatter(FuncFormatter(ssimFormatter))
            else:
                # For RSE and RAE, use log scale
                ax.set_yscale('log', base=2)

                # Custom y-axis formatter: display decimal values instead of scientific notation
                def yAxisFormatter(value, pos):
                    if value >= 1:
                        return f'{value:.0f}'
                    elif value >= 0.01:
                        return f'{value:.2f}'
                    elif value >= 0.001:
                        return f'{value:.3f}'
                    else:
                        return f'{value:.4f}'

                ax.yaxis.set_major_formatter(FuncFormatter(yAxisFormatter))

            # Set x-axis ticks to show actual SPP values
            available_spp = sorted(set([spp for method in self.methods
                                       for spp in results[metric].get(method, {}).keys()
                                       if results[metric][method][spp] is not None]))
            if available_spp:
                ax.set_xticks(available_spp)
                ax.set_xticklabels([str(spp) for spp in available_spp])
                ax.set_xlim(min(available_spp) * 0.8, max(available_spp) * 1.2)

        plt.tight_layout()
        savePath = os.path.join(self.resultDir, f'{self.sceneName}_metrics_comparison.png')
        plt.savefig(savePath, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {savePath}")
        plt.show()

