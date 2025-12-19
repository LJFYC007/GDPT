import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from Poisson import PoissonReconstructor
from GeneralizedPoisson import GeneralizedPoissonReconstructor
from jbf import GradientVarianceFilter
from Metrics import ImageComparator

cv2.setLogLevel(0) # Suppress OpenCV warnings

SppValues = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
Methods = ["pt", "poisson", "baseline", "ours"]
SceneName = "kitchen"
PathToData = f"E:/GDPT/reconstruction/data/{SceneName}"
PathToResult = f"E:/GDPT/reconstruction/result/{SceneName}"
PathToReference = f"E:/GDPT/reconstruction/data/{SceneName}/gt.exr"

class ReconstructionProcessor:
    def __init__(self, dataDir=PathToData, resultDir=PathToResult, verbose=False):
        self.dataDir = dataDir
        self.resultDir = resultDir

        self.poissonRecon = PoissonReconstructor(lambd=0.1, verbose=verbose)
        self.genPoissonRecon = GeneralizedPoissonReconstructor(eps=1e-6, verbose=verbose)

    def loadData(self, spp, method):
        """Load data for non-ours methods (baseline, poisson, pt)."""
        data = {}
        if method == "ours":
            folder = f"{spp}_ours"
        else:
            folder = f"{spp}_baseline"

        data['pt'] = cv2.imread(f"{self.dataDir}/{folder}/color.exr", cv2.IMREAD_UNCHANGED)
        if data['pt'] is not None:
            data['pt'] = data['pt'][:, :, :3]
        data['variance'] = cv2.imread(f"{self.dataDir}/{folder}/variance.exr", cv2.IMREAD_UNCHANGED)
        if data['variance'] is not None:
            data['variance'] = data['variance'][:, :, :3]
        data['gradX'] = cv2.imread(f"{self.dataDir}/{folder}/gradient_x.exr", cv2.IMREAD_UNCHANGED)
        if data['gradX'] is not None:
            data['gradX'] = data['gradX'][:, :, :3]
        data['gradY'] = cv2.imread(f"{self.dataDir}/{folder}/gradient_y.exr", cv2.IMREAD_UNCHANGED)
        if data['gradY'] is not None:
            data['gradY'] = data['gradY'][:, :, :3]
        data['varX'] = cv2.imread(f"{self.dataDir}/{folder}/gradient_x_variance.exr", cv2.IMREAD_UNCHANGED)
        if data['varX'] is not None:
            data['varX'] = data['varX'][:, :, :3]
        data['varY'] = cv2.imread(f"{self.dataDir}/{folder}/gradient_y_variance.exr", cv2.IMREAD_UNCHANGED)
        if data['varY'] is not None:
            data['varY'] = data['varY'][:, :, :3]
        data['dGradX'] = cv2.imread(f"{self.dataDir}/{folder}/degenerate_gradient_x.exr", cv2.IMREAD_UNCHANGED)
        if data['dGradX'] is not None:
            data['dGradX'] = data['dGradX'][:, :, :3]
        data['dGradY'] = cv2.imread(f"{self.dataDir}/{folder}/degenerate_gradient_y.exr", cv2.IMREAD_UNCHANGED)
        if data['dGradY'] is not None:
            data['dGradY'] = data['dGradY'][:, :, :3]
        data['dVarX'] = cv2.imread(f"{self.dataDir}/{folder}/degenerate_gradient_x_variance.exr", cv2.IMREAD_UNCHANGED)
        if data['dVarX'] is not None:
            data['dVarX'] = data['dVarX'][:, :, :3]
        data['dVarY'] = cv2.imread(f"{self.dataDir}/{folder}/degenerate_gradient_y_variance.exr", cv2.IMREAD_UNCHANGED)
        if data['dVarY'] is not None:
            data['dVarY'] = data['dVarY'][:, :, :3]

        data['rawVarX'] = cv2.imread(f"{self.dataDir}/{folder}/gradient_x_variance_raw.exr", cv2.IMREAD_UNCHANGED)
        if data['rawVarX'] is not None:
            data['rawVarX'] = data['rawVarX'][:, :, :3]
        data['rawVarY'] = cv2.imread(f"{self.dataDir}/{folder}/gradient_y_variance_raw.exr", cv2.IMREAD_UNCHANGED)
        if data['rawVarY'] is not None:
            data['rawVarY'] = data['rawVarY'][:, :, :3]
        data['dRawVarX'] = cv2.imread(f"{self.dataDir}/{folder}/degenerate_gradient_x_variance_raw.exr", cv2.IMREAD_UNCHANGED)
        if data['dRawVarX'] is not None:
            data['dRawVarX'] = data['dRawVarX'][:, :, :3]
        data['dRawVarY'] = cv2.imread(f"{self.dataDir}/{folder}/degenerate_gradient_y_variance_raw.exr", cv2.IMREAD_UNCHANGED)
        if data['dRawVarY'] is not None:
            data['dRawVarY'] = data['dRawVarY'][:, :, :3]
        return data

    def saveResult(self, result, method, spp):
        methodDir = f"{self.resultDir}/{method}"
        os.makedirs(methodDir, exist_ok=True)
        path = f"{methodDir}/{method}_{spp}.exr"
        cv2.imwrite(path, result.astype(np.float32))
        print(f"Successfully reconstructed and saved {method} reconstruction with {spp}-spp")

    def runPoisson(self, spp):
        data = self.loadData(spp, "poisson")
        if all(data[k] is not None for k in ['pt', 'gradX', 'gradY', 'dGradX', 'dGradY']):
            result = self.poissonRecon.reconstruct(data['gradY'] + data['dGradY'], data['gradX'] + data['dGradX'], data['pt'])
            self.saveResult(result, "poisson", spp)
            return result
        return None

    def runBaseline(self, spp):
        data = self.loadData(spp, "baseline")
        required = ['pt', 'gradX', 'gradY', 'variance', 'varX', 'varY', 'dGradX', 'dGradY', 'dVarX', 'dVarY']
        if all(data[k] is not None for k in required):
            result = self.genPoissonRecon.reconstruct(
                data['pt'], data['variance'],
                data['gradX'] + data['dGradX'], data['gradY'] + data['dGradY'],
                data['varX'] + data['dVarX'], data['varY'] + data['dVarY'])
            self.saveResult(result, "baseline", spp)
            return result
        return None

    def runOurs(self, spp, epsilon=1e-6):
        data = self.loadData(spp, "ours")
        required = ['pt', 'gradX', 'gradY', 'variance', 'rawVarX', 'rawVarY', 'dGradX', 'dGradY', 'dRawVarX', 'dRawVarY']

        if all(data[k] is not None for k in required):
            m2X = data['dGradX'] * data['dGradX']
            m2Y = data['dGradY'] * data['dGradY']

            gammaX = m2X / (m2X + data['dRawVarX'] / spp + epsilon)
            gammaY = m2Y / (m2Y + data['dRawVarY'] / spp + epsilon)

            # Blend grad and dGrad using gamma weights
            blendedGradX = gammaX * data['dGradX'] + data['gradX']
            blendedGradY = gammaY * data['dGradY'] + data['gradY']

            blendedVarX = data['rawVarX'] + (gammaX * gammaX) * data['dRawVarX']
            blendedVarY = data['rawVarY'] + (gammaY * gammaY) * data['dRawVarY']

            # Save blended gradients and raw variances for debugging
            folder = f"{spp}_ours"
            folderPath = os.path.join(self.dataDir, folder)
            os.makedirs(folderPath, exist_ok=True)
            cv2.imwrite(os.path.join(folderPath, "blendedGradX.exr"), blendedGradX.astype(np.float32))
            cv2.imwrite(os.path.join(folderPath, "blendedGradY.exr"), blendedGradY.astype(np.float32))
            cv2.imwrite(os.path.join(folderPath, "blendedVarX_raw.exr"), blendedVarX.astype(np.float32))
            cv2.imwrite(os.path.join(folderPath, "blendedVarY_raw.exr"), blendedVarY.astype(np.float32))

            # Apply JBF to blended variances
            varianceFilter = GradientVarianceFilter(os.path.dirname(self.dataDir), SceneName)
            filteredVarX, filteredVarY = varianceFilter.filterGradientVariance(blendedVarX, blendedVarY)

            # Save filtered variances
            cv2.imwrite(os.path.join(folderPath, "blendedVarX.exr"), filteredVarX.astype(np.float32))
            cv2.imwrite(os.path.join(folderPath, "blendedVarY.exr"), filteredVarY.astype(np.float32))

            result = self.genPoissonRecon.reconstruct(
                data['pt'], data['variance'],
                blendedGradX, blendedGradY,
                filteredVarX, filteredVarY)
            self.saveResult(result, "ours", spp)
            return result
        return None

    def runAll(self, methods, sppList):
        for spp in sppList:
            data = self.loadData(spp, "pt")
            if data['pt'] is not None:
                self.saveResult(data['pt'], "pt", spp)
            if "poisson" in methods:
                self.runPoisson(spp)
            if "baseline" in methods:
                self.runBaseline(spp)
            if "ours" in methods:
                self.runOurs(spp)


if __name__ == "__main__":
    processor = ReconstructionProcessor(verbose=False)
    # processor.runAll(Methods, SppValues)

    comparator = ImageComparator(PathToResult, PathToReference, Methods, SppValues, SceneName)
    results = comparator.compareWithReference()
