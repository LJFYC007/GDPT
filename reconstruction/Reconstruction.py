import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from Poisson import PoissonReconstructor
from GeneralizedPoisson import GeneralizedPoissonReconstructor

cv2.setLogLevel(0) # Suppress OpenCV warnings

SppValues = [32, 64, 128, 1024, 50000]
Methods = ["pt", "poisson", "generalized-poisson"]

class ImageComparator:
    def __init__(self, resultDir="../minimal_result", refFile="reference-staircase.exr"):
        self.resultDir = resultDir
        self.refFile = refFile

    def calculateMSE(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    def compareWithReference(self):
        refPath = os.path.join(self.resultDir, self.refFile)
        refImg = cv2.imread(refPath, cv2.IMREAD_UNCHANGED)
        if refImg is None:
            raise FileNotFoundError(f"Cannot read: {refPath}")

        results = {}
        for method in Methods:
            results[method] = {}
            for spp in SppValues:
                imgPath = os.path.join(self.resultDir, f"{method}-{spp}.exr")
                img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

                if img is not None:
                    results[method][spp] = self.calculateMSE(img, refImg)
                else:
                    results[method][spp] = None

        self.printResults(results)
        return results

    def printResults(self, results):
        print("\n=== MSE Results ===")
        header = "Method".ljust(12)
        for spp in SppValues:
            header += f"{spp}spp".ljust(15)
        print(header)
        print("-" * len(header))

        for method in Methods:
            row = method.ljust(12)
            for spp in SppValues:
                mse = results.get(method, {}).get(spp)
                if mse is not None:
                    row += f"{mse:.6f}".ljust(15)
                else:
                    row += " ".ljust(15)
            print(row)
        print()


class ReconstructionProcessor:
    def __init__(self, outputDir="../output", resultDir="../minimal_result", verbose=False):
        self.outputDir = outputDir
        self.resultDir = resultDir

        self.poissonRecon = PoissonReconstructor(lambd=0.1, verbose=verbose)
        self.genPoissonRecon = GeneralizedPoissonReconstructor(eps=1e-6, verbose=verbose)

    def loadData(self, spp, method="poisson"):
        data = {}
        data['pt'] = cv2.imread(f"{self.outputDir}/Mogwai.AccumulatePass.output.{spp}.exr", cv2.IMREAD_UNCHANGED)
        data['gradX'] = cv2.imread(f"{self.outputDir}/Mogwai.ErrorMeasureXPass.Output.{spp}.exr", cv2.IMREAD_UNCHANGED)
        data['gradY'] = cv2.imread(f"{self.outputDir}/Mogwai.ErrorMeasureYPass.Output.{spp}.exr", cv2.IMREAD_UNCHANGED)

        if method == "generalized-poisson":
            data['variance'] = cv2.imread(f"{self.outputDir}/Mogwai.AccumulatePass.variance.{spp}.exr", cv2.IMREAD_UNCHANGED)
            data['varX'] = cv2.imread(f"{self.outputDir}/Mogwai.AccumulatePassX.variance.{spp}.exr", cv2.IMREAD_UNCHANGED)
            data['varY'] = cv2.imread(f"{self.outputDir}/Mogwai.AccumulatePassY.variance.{spp}.exr", cv2.IMREAD_UNCHANGED)

        return data

    def saveResult(self, result, method, spp):
        path = f"{self.resultDir}/{method}-{spp}.exr"
        cv2.imwrite(path, result.astype(np.float32))
        print(f"Successfully reconstructed and saved {method} reconstruction with {spp}-spp")

    def runPoisson(self, spp):
        data = self.loadData(spp)
        if all(data[k] is not None for k in ['pt', 'gradX', 'gradY']):
            result = self.poissonRecon.reconstruct(data['gradY'], data['gradX'], data['pt'])
            self.saveResult(result, "poisson", spp)
            return result
        return None

    def runGeneralizedPoisson(self, spp):
        data = self.loadData(spp, "generalized-poisson")
        required = ['pt', 'gradX', 'gradY', 'variance', 'varX', 'varY']
        if all(data[k] is not None for k in required):
            result = self.genPoissonRecon.reconstruct(
                data['pt'], data['variance'],
                data['gradX'], data['gradY'],
                data['varX'], data['varY'])
            self.saveResult(result, "generalized-poisson", spp)
            return result
        return None

    def runAll(self, methods, sppList):
        for spp in sppList:
            data = self.loadData(spp)
            if data['pt'] is not None:
                self.saveResult(data['pt'], "pt", spp)

            if "poisson" in methods:
                self.runPoisson(spp)
            if "generalized-poisson" in methods:
                self.runGeneralizedPoisson(spp)


if __name__ == "__main__":
    processor = ReconstructionProcessor()
    processor.runAll(["poisson"], SppValues)

    comparator = ImageComparator()
    results = comparator.compareWithReference()
