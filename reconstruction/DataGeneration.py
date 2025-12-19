import os
import shutil
import argparse
from typing import List

import yaml
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from jbf import GradientVarianceFilter

# File mappings for render outputs
RENDER_OUTPUT_FILES = [
    ("color.exr", "variance.exr"),
    ("gradient_x.exr", "gradient_y.exr"),
    ("gradient_x_variance.exr", "gradient_y_variance.exr"),
    ("degenerate_gradient_x.exr", "degenerate_gradient_y.exr"),
    ("degenerate_gradient_x_variance.exr", "degenerate_gradient_y_variance.exr"),
]

GRADIENT_VARIANCE_FILES = [
    ('gradient_x_variance.exr', 'gradient_x'),
    ('gradient_y_variance.exr', 'gradient_y'),
    ('degenerate_gradient_x_variance.exr', 'gradient_x'),
    ('degenerate_gradient_y_variance.exr', 'gradient_y')
]


def parseSceneTemplate(templatePath: str, spp: int, resolution: List[int],
                        seed: int = 0) -> str:
    """Parse scene template file and replace placeholder variables."""
    with open(templatePath, 'r') as f:
        sceneText = f.read()

    replacements = {
        "$$SPP$$": str(spp),
        "$$RES_X$$": str(resolution[0]),
        "$$RES_Y$$": str(resolution[1]),
        "$$SEED$$": str(seed),
        "$$FILE$$": f"{spp}_temp.exr"
    }

    for placeholder, value in replacements.items():
        sceneText = sceneText.replace(placeholder, value)

    return sceneText


def applyJbfToVariances(expname: str, spp: int, outputDir: str, targetDir: str) -> None:
    """Apply joint bilateral filter to variance images using gradient+color guide."""
    print(f"\nApplying JBF to variances for {expname}, SPP={spp}...")

    # Initialize variance filter
    varianceFilter = GradientVarianceFilter(outputDir, expname)

    # Filter variance.exr
    variancePath = os.path.join(targetDir, "variance.exr")
    if os.path.exists(variancePath):
        variance = cv2.imread(variancePath, cv2.IMREAD_UNCHANGED)[:, :, :3]
        filtered = varianceFilter.filterColorVariance(variance)
        cv2.imwrite(variancePath, filtered.astype(np.float32))
        print(f"  Filtered: variance.exr")

    # Filter gradient variances
    gradXVar = cv2.imread(os.path.join(targetDir, "gradient_x_variance.exr"), cv2.IMREAD_UNCHANGED)
    gradYVar = cv2.imread(os.path.join(targetDir, "gradient_y_variance.exr"), cv2.IMREAD_UNCHANGED)
    dGradXVar = cv2.imread(os.path.join(targetDir, "degenerate_gradient_x_variance.exr"), cv2.IMREAD_UNCHANGED)
    dGradYVar = cv2.imread(os.path.join(targetDir, "degenerate_gradient_y_variance.exr"), cv2.IMREAD_UNCHANGED)

    if gradXVar is not None and gradYVar is not None:
        gradXVar, gradYVar = gradXVar[:, :, :3], gradYVar[:, :, :3]
        filteredGradXVar, filteredGradYVar = varianceFilter.filterGradientVariance(gradXVar, gradYVar)
        cv2.imwrite(os.path.join(targetDir, "gradient_x_variance.exr"), filteredGradXVar.astype(np.float32))
        cv2.imwrite(os.path.join(targetDir, "gradient_y_variance.exr"), filteredGradYVar.astype(np.float32))
        print(f"  Filtered: gradient_x_variance.exr, gradient_y_variance.exr")

    if dGradXVar is not None and dGradYVar is not None:
        dGradXVar, dGradYVar = dGradXVar[:, :, :3], dGradYVar[:, :, :3]
        filteredDGradXVar, filteredDGradYVar = varianceFilter.filterGradientVariance(dGradXVar, dGradYVar)
        cv2.imwrite(os.path.join(targetDir, "degenerate_gradient_x_variance.exr"), filteredDGradXVar.astype(np.float32))
        cv2.imwrite(os.path.join(targetDir, "degenerate_gradient_y_variance.exr"), filteredDGradYVar.astype(np.float32))
        print(f"  Filtered: degenerate_gradient_x_variance.exr, degenerate_gradient_y_variance.exr")

def findSceneTemplate(expname: str) -> str:
    """Find the scene template file for a given experiment name."""
    for ext in ['.luisa', '.json']:
        path = f"E:/UnbiasedGD/scene/{expname}/scene-template{ext}"
        if os.path.exists(path): return path
    raise FileNotFoundError(f"Scene template not found for experiment: {expname}")


def copyRenderOutputs(sceneDir: str, outputDir: str, spp: int) -> None:
    """Copy rendered files from scene directory to output directory."""
    files = sum(RENDER_OUTPUT_FILES, ())
    for dstName in files:
        srcName = f"{spp}_temp.exr" if dstName == "color.exr" else f"{spp}_temp_{dstName}"
        srcPath = os.path.join(sceneDir, srcName)
        if os.path.exists(srcPath):
            shutil.copy2(srcPath, os.path.join(outputDir, dstName))
            print(f"  Copied: {dstName}")
        else:
            print(f"  Warning: {srcName} not found")


def cleanupTempFiles(sceneDir: str, spp: int) -> None:
    """Remove temporary files generated during rendering."""
    for suffix in ["gradient_x", "gradient_y", "degenerate_gradient_x", "degenerate_gradient_y"]:
        filePath = os.path.join(sceneDir, f"{spp}_temp_{suffix}_single_frame.exr")
        if os.path.exists(filePath): os.remove(filePath)


def renderSceneOnce(expname: str, spp: int, config: dict, sceneDir: str) -> str:
    """Render a scene once and return the temporary output directory."""
    resolution = config.get("scene-params", {}).get("resolution", [1280, 720])

    templatePath = findSceneTemplate(expname)
    sceneText = parseSceneTemplate(templatePath, spp, resolution, seed=0)
    tempSceneFile = os.path.join(sceneDir, f"{spp}_temp.luisa")

    with open(tempSceneFile, 'w') as f: f.write(sceneText)

    renderCmd = f"{config['luisa-render']['path']} -b {config['luisa-render']['backend']} {tempSceneFile}"
    print(f"  Command: {renderCmd}")
    os.system(renderCmd)

    os.remove(tempSceneFile)
    cleanupTempFiles(sceneDir, spp)
    return sceneDir


def processBaselineAndOurs(expname: str, spp: int, config: dict, outputBase: str) -> None:
    """Render once and create baseline and ours versions with different post-processing."""
    print(f"\n  Rendering SPP={spp}...")

    sceneDir = os.path.dirname(findSceneTemplate(expname))
    renderSceneOnce(expname, spp, config, sceneDir)

    baselineDir = os.path.join(outputBase, expname, f"{spp}_baseline")
    oursDir = os.path.join(outputBase, expname, f"{spp}_ours")
    os.makedirs(baselineDir, exist_ok=True)
    os.makedirs(oursDir, exist_ok=True)

    copyRenderOutputs(sceneDir, baselineDir, spp)
    copyRenderOutputs(sceneDir, oursDir, spp)

    # Clean up temp files in scene directory
    files = sum(RENDER_OUTPUT_FILES, ())
    for dstName in files:
        srcName = f"{spp}_temp.exr" if dstName == "color.exr" else f"{spp}_temp_{dstName}"
        tempPath = os.path.join(sceneDir, srcName)
        if os.path.exists(tempPath): os.remove(tempPath)

    # Baseline: Apply JBF to all variances
    applyJbfToVariances(expname, spp, outputBase, baselineDir)

    # Ours: Apply JBF to variance.exr, rename gradient variances to _raw
    print(f"\n  Processing ours version...")

    varianceFilter = GradientVarianceFilter(outputBase, expname)

    variancePath = os.path.join(oursDir, "variance.exr")
    if os.path.exists(variancePath):
        variance = cv2.imread(variancePath, cv2.IMREAD_UNCHANGED)[:, :, :3]
        filtered = varianceFilter.filterColorVariance(variance)
        cv2.imwrite(variancePath, filtered.astype(np.float32))
        print(f"  Filtered: variance.exr")

    for filename, _ in GRADIENT_VARIANCE_FILES:
        filePath = os.path.join(oursDir, filename)
        if not os.path.exists(filePath):
            print(f"  Warning: {filename} not found, skipping")
            continue
        rawPath = os.path.join(oursDir, filename.replace(".exr", "_raw.exr"))
        if os.path.exists(rawPath): os.remove(rawPath)
        shutil.move(filePath, rawPath)
        print(f"  Renamed to raw: {os.path.basename(rawPath)}")


def main():
    """Main entry point for the rendering script."""
    parser = argparse.ArgumentParser(description="Render scenes with specified SPP values and JBF filtering")
    parser.add_argument("-s", "--spps", type=int, nargs="+", required=True, help="List of SPP values")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("-e", "--expname", type=str, nargs="+", required=True, help="Experiment names")
    parser.add_argument("-c", "--config", type=str, default="render.yaml", help="Config file (default: render.yaml)")
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    for expname in args.expname:
        print(f"\n{'='*60}\nProcessing experiment: {expname}\n{'='*60}")
        try:
            for spp in args.spps: processBaselineAndOurs(expname, spp, config, args.output)
            print(f"\n✓ Completed: {expname}")
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            continue

    print(f"\n{'='*60}\nAll renders completed\n{'='*60}")


if __name__ == "__main__":
    main()
