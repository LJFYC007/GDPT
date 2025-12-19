import numpy as np
import imageio.v2 as imageio
import os
import OpenEXR
import Imath

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def joint_bilateral_filter(input_img, guide_img, d=9, sigma_color=0.1, sigma_space=75):
    """
    Apply joint bilateral filter to input image using guide image with vectorized operations.

    Args:
        input_img: Input image to filter (HDR, float32, shape: HxWx3)
        guide_img: Guidance image (HDR, float32, shape: HxWxC)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image (same shape as input_img)
    """
    height, width, channels = input_img.shape

    # Normalize guide image
    guide_max = np.max(np.abs(guide_img)) or 1.0
    guide_norm = guide_img / guide_max

    # Create Gaussian spatial kernel
    radius = d // 2
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

    # Pad arrays
    pad = radius
    input_padded = np.pad(input_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    guide_padded = np.pad(guide_norm, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    filtered = np.zeros_like(input_img, dtype=np.float32)

    # Reshape for vectorized processing
    kernel_size = d
    patches_shape = (height, width, kernel_size, kernel_size)

    # Extract patches using stride_tricks
    guide_patches = np.lib.stride_tricks.sliding_window_view(
        guide_padded, (kernel_size, kernel_size, guide_padded.shape[-1])
    )[:height, :width].reshape(height, width, kernel_size, kernel_size, -1)

    # Center pixels for each patch
    center_pixels = guide_norm[:, :, np.newaxis, np.newaxis, :]

    # Compute color differences
    color_diff = np.sqrt(np.sum((guide_patches - center_pixels)**2, axis=-1))
    color_kernel = np.exp(-(color_diff**2) / (2 * sigma_color**2))

    # Combine kernels
    kernel = spatial_kernel * color_kernel
    kernel_sum = np.sum(kernel, axis=(2, 3), keepdims=True)
    kernel_sum = np.where(kernel_sum == 0, 1.0, kernel_sum)
    kernel /= kernel_sum

    # Apply to each channel
    for c in range(channels):
        input_patches = np.lib.stride_tricks.sliding_window_view(
            input_padded[:, :, c], (kernel_size, kernel_size)
        )[:height, :width]
        filtered[:, :, c] = np.sum(input_patches * kernel, axis=(2, 3))

    return filtered


def readExr(path):
    """Read an EXR file and return it as a numpy array."""
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    floatType = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []
    for channelName in ['R', 'G', 'B']:
        channelData = np.frombuffer(
            exr.channel(channelName, floatType),
            dtype=np.float32
        ).reshape(height, width)
        channels.append(channelData)

    return np.stack(channels, axis=-1)


def writeExr(image, outputPath):
    """Write a numpy array to an EXR file."""
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    imageio.imwrite(outputPath, image.astype(np.float32), format='EXR')


def computeGradients(image):
    """Compute x and y gradients of an image."""
    gradX = np.zeros_like(image, dtype=np.float64)
    gradY = np.zeros_like(image, dtype=np.float64)
    gradX[:, :-1, :] = image[:, 1:, :] - image[:, :-1, :]
    gradX[:, -1, :] = 0
    gradY[:-1, :, :] = image[1:, :, :] - image[:-1, :, :]
    gradY[-1, :, :] = 0
    return gradX, gradY


def computeAndSaveGradients(featureDir, featureName):
    """Compute and save gradients for a given feature if they don't exist."""
    gradXPath = os.path.join(featureDir, f"{featureName}_grad_x.exr")
    if os.path.exists(gradXPath): return

    featureImg = readExr(os.path.join(featureDir, f"{featureName}.exr"))
    gradX, gradY = computeGradients(featureImg)
    writeExr(gradX, gradXPath)
    writeExr(gradY, os.path.join(featureDir, f"{featureName}_grad_y.exr"))


class GradientVarianceFilter:
    """Helper class to apply JBF to gradient variances using precomputed guide images."""

    def __init__(self, dataDir, sceneName, d=11, sigmaColor=0.01, sigmaSpace=3):
        self.dataDir = dataDir
        self.sceneName = sceneName
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.guideImages = None
        self.featureDir = None

    def loadGuideImages(self):
        """Load and construct guide images from feature data."""
        self.featureDir = os.path.join(self.dataDir, self.sceneName, "feature")

        # Load features
        normal = cv2.imread(os.path.join(self.featureDir, "normal.exr"), cv2.IMREAD_UNCHANGED)[:, :, :3]
        albedo = cv2.imread(os.path.join(self.featureDir, "albedo.exr"), cv2.IMREAD_UNCHANGED)[:, :, :3]

        # Compute or load gradients
        normalGradXPath = os.path.join(self.featureDir, "normal_grad_x.exr")
        normalGradYPath = os.path.join(self.featureDir, "normal_grad_y.exr")
        albedoGradXPath = os.path.join(self.featureDir, "albedo_grad_x.exr")
        albedoGradYPath = os.path.join(self.featureDir, "albedo_grad_y.exr")

        if os.path.exists(normalGradXPath):
            normalGradX = cv2.imread(normalGradXPath, cv2.IMREAD_UNCHANGED)[:, :, :3]
            normalGradY = cv2.imread(normalGradYPath, cv2.IMREAD_UNCHANGED)[:, :, :3]
        else:
            normalGradX, normalGradY = computeGradients(normal)
            cv2.imwrite(normalGradXPath, normalGradX.astype(np.float32))
            cv2.imwrite(normalGradYPath, normalGradY.astype(np.float32))

        if os.path.exists(albedoGradXPath):
            albedoGradX = cv2.imread(albedoGradXPath, cv2.IMREAD_UNCHANGED)[:, :, :3]
            albedoGradY = cv2.imread(albedoGradYPath, cv2.IMREAD_UNCHANGED)[:, :, :3]
        else:
            albedoGradX, albedoGradY = computeGradients(albedo)
            cv2.imwrite(albedoGradXPath, albedoGradX.astype(np.float32))
            cv2.imwrite(albedoGradYPath, albedoGradY.astype(np.float32))

        # Construct guide images
        self.guideImages = {
            'color': np.concatenate((normal, albedo), axis=-1),
            'gradientX': np.concatenate((normalGradX, normal, albedoGradX, albedo), axis=-1),
            'gradientY': np.concatenate((normalGradY, normal, albedoGradY, albedo), axis=-1)
        }

    def filterColorVariance(self, variance):
        """Apply JBF to color variance using normal+albedo guide."""
        if self.guideImages is None:
            self.loadGuideImages()

        return joint_bilateral_filter(variance, self.guideImages['color'],
                                     d=self.d, sigma_color=self.sigmaColor, sigma_space=self.sigmaSpace)

    def filterGradientVariance(self, varX, varY):
        """Apply JBF to gradient variances and return filtered results."""
        if self.guideImages is None:
            self.loadGuideImages()

        filteredVarX = joint_bilateral_filter(varX, self.guideImages['gradientX'],
                                             d=self.d, sigma_color=self.sigmaColor, sigma_space=self.sigmaSpace)
        filteredVarY = joint_bilateral_filter(varY, self.guideImages['gradientY'],
                                             d=self.d, sigma_color=self.sigmaColor, sigma_space=self.sigmaSpace)

        return filteredVarX, filteredVarY
