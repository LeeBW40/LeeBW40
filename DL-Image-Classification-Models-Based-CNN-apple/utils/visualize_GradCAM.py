import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision import datasets, transforms

from classic_models.googlenet_v1 import GoogLeNet
from classic_models.alexnet import AlexNet
from classic_models.mobilenet_v3 import MobileNetV3,mobilenet_v3_large
from classic_models.mobilenet_v3pro import mobilenet_v3_large_CBAM,MobileNetV3_CBAM  # "mobilenetV3" + "mobilenetV3+CBAM" + "mobileNetV3+Ghost"
from classic_models.mobilenet_v3_ghost import mobilenet_v3_large_ghost
from classic_models.resnet import ResNet,resnet101

"""
可视化模型在分类时关注的图像区域
"""
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # 注册前向和反向钩子
        fwd_handle = self.target_layer.register_forward_hook(self.save_activations)
        bwd_handle = self.target_layer.register_backward_hook(self.save_gradients)
        self.hook_handles.extend([fwd_handle, bwd_handle])


    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        # 前向传播
        output = self.model(x)

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 生成CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # 后处理
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam

    def __del__(self):
        # 移除钩子
        for handle in self.hook_handles:
            handle.remove()


class GradCAM_new:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 使用新版hook注册方式
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        # 前向传播
        output = self.model(x)

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 生成CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # 后处理
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = np.clip(cam, 0, 1)
        return cam

    def __del__(self):
        # 移除钩子
        for handle in self.hook_handles:
            handle.remove()


#+++++++++++++++++++++++++++++++++++=图片格式处理++++++++++++++++++++++++++++++++++
# 支持的图片格式
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_valid_image(filepath):
    """验证是否为真实图片文件"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # 验证文件完整性
        return True
    except Exception as e:
        print(f"无效图片文件: {filepath} - {str(e)}")
        return False


def process_images(root_dir):
    """统一处理文件夹中的图片类型"""
    valid_images = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 统一小写扩展名
            base, ext = os.path.splitext(file)
            ext = ext.lower()

            # 过滤非图片文件
            if ext not in VALID_EXTENSIONS:
                continue

            full_path = os.path.join(root, file)

            # 验证是否为真实图片
            if is_valid_image(full_path):
                valid_images.append(full_path)

    return valid_images


def safe_imread(img_path):
    """兼容中文路径的图片读取"""
    try:
        # 处理Windows中文路径编码
        img_path_gbk = img_path.encode('gbk',errors='ignore')
        # 通过numpy读取二进制文件
        img_array = np.fromfile(img_path_gbk, dtype=np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        raise ValueError("OpenCV无法解码该文件")

    except Exception as e:
        print(f"读取失败: {img_path} - {str(e)}")
        return None


def overlay_heatmap(img, heatmap):
    # 确保热力图为三通道
    if len(heatmap.shape) == 2:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

    # 统一尺寸
    if img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 验证数据类型
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if heatmap.dtype != np.uint8:
        heatmap = (heatmap * 255).astype(np.uint8)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


def visualize_gradcam(model, device, data_path, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果保存至:{os.path.abspath(save_dir)}")

    # 加载预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取验证集数据
    val_dir = os.path.join(data_path, 'val')
    # test_images = process_image(val_dir)
    test_images = process_images(val_dir)

    # random.seed(42)
    # # 随机展示五张图
    # test_images = random.sample(test_images, min(5, len(test_images))) if len(test_images) > 0 else []
    #
    # print(f"找到 {len(test_images)} 张测试图像，示例路径：{test_images[:1]}")

    # 确定目标层（关键！需匹配您的模型结构）

    if not test_images:
        print("未找到测试图片，请补充数据")
        return

    print(f"共找到 {len(test_images)} 张测试图片，开始处理...")

    if isinstance(model, AlexNet):
        target_layer = model.features[10]     # alexnet的最后一个卷积层
    else:
        target_layer = model.features[-1][-1].conv  # MobileNetV3的最后一个卷积块

    # 初始化Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # for img_path in test_images:
    #     # img = robust_image_reader(img_path)
    #     img = safe_imread(img_path)
    #     if img is None:
    #         continue
    #
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     # 预处理
    #     input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    #
    #     # 预测
    #     with torch.no_grad():
    #         output = model(input_tensor)
    #     pred_class = torch.argmax(output).item()
    #
    #     # 生成CAM
    #     cam = gradcam.generate(input_tensor, pred_class)
    #
    #     # 可视化叠加
    #     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    #     # superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    #     superimposed = overlay_heatmap(img, heatmap=heatmap)
    #     # 保存结果
    #     fig = plt.figure(figsize=(12, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img_rgb)
    #     plt.title('Original')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    #     plt.title('Grad-CAM')
    #
    #     plt.savefig(os.path.join(save_dir, os.path.basename(img_path)))
    #     plt.close()

    # 处理每张图片
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n处理进度: [{idx}/{len(test_images)}] - {os.path.basename(img_path)}")

        try:
            # 读取图片（兼容中文路径）
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("图片读取失败")

            # 预处理
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

            # 预测类别
            with torch.no_grad():
                output = model(input_tensor)
            pred_class = torch.argmax(output).item()

            # 生成热力图
            cam = gradcam.generate(input_tensor, pred_class)

            # 生成叠加图像
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            superimposed = overlay_heatmap(img, heatmap=heatmap)
            # 保存结果
            fig = plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title('Original')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
            plt.title('Grad-CAM')

            plt.savefig(os.path.join(save_dir, os.path.basename(img_path)))
            plt.close()

            # # 保存结果（保留原始文件名）
            # base_name = os.path.basename(img_path)
            # save_path = os.path.join(save_dir, f"gradcam_{base_name}")
            # cv2.imwrite(save_path, superimposed)
            # print(f"已保存: {save_path}")

        except Exception as e:
            print(f"处理失败: {os.path.basename(img_path)} - {str(e)}")
            continue

    print("\n所有图片处理完成！")

