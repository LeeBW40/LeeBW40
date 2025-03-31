from PIL import ImageFilter, Image
import cv2
import numpy as np


class GaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class MeanFilter:
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, img):
        # return img.filter(ImageFilter.Kernel((3,3), [1]*9, scale=9))
        return img.filter(ImageFilter.BoxBlur(self.radius))


class MedianFilter:
    def __init__(self, size=3):
        self.size = size

    def __call__(self, img):
        return img.filter(ImageFilter.MedianFilter(size=self.size))

# 这将Sobel算子的结果转换为灰度，然后合并回RGB。但这样会将图像转换为边缘检测后的灰度图，可能改变图像的颜色信息，但作为数据增强，可能可以接受。
class SobelFilter:
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
        sobel_mag = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(sobel_mag)


class LaplaceFilter:
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplace = cv2.Laplacian(gray, cv2.CV_64F)
        laplace = np.uint8(255 * np.abs(laplace) / np.max(np.abs(laplace)))
        laplace = cv2.cvtColor(laplace, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(laplace)


# 这样处理后，得到的是边缘强度图像，转换为RGB格式。这可能更适合作为数据增强，尽管颜色信息被丢失，但作为数据增强的手段，可以增加模型对边缘特征的鲁棒性
class RobertFilter:
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        robert_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        robert_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rob_x = cv2.filter2D(gray, -1, robert_x)
        rob_y = cv2.filter2D(gray, -1, robert_y)
        rob_mag = np.sqrt(rob_x**2 + rob_y**2)
        rob_mag = np.uint8(255 * rob_mag / np.max(rob_mag))
        rob_mag = cv2.cvtColor(rob_mag, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rob_mag)