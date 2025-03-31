import os
import sys
import torch
import json
import time
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from datetime import datetime
from classic_models.googlenet_v1 import GoogLeNet
from classic_models.alexnet import AlexNet
from  classic_models.mobilenet_v3 import MobileNetV3,mobilenet_v3_large
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.FIller import *

import os
import sys
import torch
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import  seaborn as sns
from utils.FIller import *


# 实验配置
class ExperimentConfig:
    def __init__(self):
        # 数据路径
        self.data_path = "./dataset/apple"
        self.class_dict_path = "./dataset/apple/class_indices.json"

        # 实验参数
        self.batch_size = 8
        self.epochs = 70
        self.lr = 0.0002
        self.loss_func = nn.CrossEntropyLoss()

        # 实验模式
        self.mode = "smooth"  # smooth/sharp/combined

        # 结果存储
        self.result_dir = "./results/experiment_results"
        os.makedirs(self.result_dir, exist_ok=True)


# 增强方法定义
class FilterLibrary:
    @staticmethod
    def get_smooth_transforms(method):
        """返回指定平滑方法的数据增强组合"""
        methods = {
            'mean': transforms.RandomApply([MeanFilter()], p=1.0),
            'median': transforms.RandomApply([MedianFilter()], p=1.0),
            'gaussian': transforms.RandomApply([GaussianBlur()], p=1.0)
        }
        return methods[method]

    @staticmethod
    def get_sharp_transforms(method):
        """返回指定锐化方法的数据增强组合"""
        methods = {
            'robert': transforms.RandomApply([RobertFilter()], p=1.0),
            'sobel': transforms.RandomApply([SobelFilter()], p=1.0),
            'laplace': transforms.RandomApply([LaplaceFilter()], p=1.0)
        }
        return methods[method]


# 实验执行器
class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_datasets()

    def _prepare_datasets(self):
        """准备实验数据集"""
        base_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准归一化
        ])

        # 根据实验模式添加增强
        if self.config.mode == "smooth":
            self.transforms = {
                method: transforms.Compose([
                    FilterLibrary.get_smooth_transforms(method),
                    base_transform
                ]) for method in ['mean', 'median', 'gaussian']
            }
        elif self.config.mode == "sharp":
            self.transforms = {
                method: transforms.Compose([
                    FilterLibrary.get_sharp_transforms(method),
                    base_transform
                ]) for method in ['robert', 'sobel', 'laplace']
            }
        elif self.config.mode == "combined":
            # 组合最佳方法示例（需根据实际结果修改）
            best_smooth = 'mean'  # 假设高斯滤波最佳
            best_sharp = 'robert'  # 假设Sobel最佳
            self.transforms = {
                'best_combo': transforms.Compose([
                    FilterLibrary.get_smooth_transforms(best_smooth),
                    FilterLibrary.get_sharp_transforms(best_sharp),
                    base_transform
                ])
            }

    def run_experiments(self):
        """执行所有实验"""
        results = []
        for method, transform in self.transforms.items():
            print(f"\n正在执行实验：{method.upper()}")

            # 数据加载
            train_set = datasets.ImageFolder(
                os.path.join(self.config.data_path, 'train'),
                transform=transform
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            # 模型训练
            model = GoogLeNet(num_classes=len(train_set.classes)).to(self.device)
            # optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
            optimizer = optim.AdamW(model.parameters(), lr=self.config.lr,weight_decay=1e-4,amsgrad=True)
            # # 学习率调度器（网页44建议）
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=self.config.epochs,
            #     eta_min=self.config.lr * 0.01
            # )

            train_loss_list = []
            val_loss_list = []
            train_acc_list = []
            val_acc_list = []
            best_acc = 0.0
            start_time = time.time()
            for epoch in range(self.config.epochs):
                # 训练过程...
                model.train()
                running_loss = 0.0  # 新增：记录每个epoch的总损失
                acc_num = torch.zeros(1).to(self.device)
                sample_num = 0  # 用于记录当前迭代中，已经计算量多少个样本

                train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
                for data in train_bar:
                    images, labels = data
                    sample_num += images.shape[0]
                    print(images.shape)

                    optimizer.zero_grad()
                    outputs = model(images.to(self.device))
                    pred_class = torch.max(outputs, dim=1)[1]
                    acc_num += torch.eq(pred_class, labels.to(self.device)).sum()

                    loss = self.config.loss_func(outputs, labels.to(self.device))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()  # 新增：累计损失值
                    # train_acc = acc_num.item() / sample_num
                    train_bar.desc = f"train epoch[{epoch + 1}/{self.config.epochs}], loss:{loss:.3f}"

                # 计算平均训练损失
                epoch_train_loss = running_loss / len(train_loader)
                train_loss_list.append(epoch_train_loss)

                # 计算训练准确率
                epoch_train_acc = acc_num.item() / sample_num
                train_acc_list.append(epoch_train_acc)
            # 验证过程...
            


            # 记录结果
                if epoch_train_acc > best_acc:
                    best_acc = epoch_train_acc

            end_time = time.time()
            training_time = end_time - start_time
            training_time = round(training_time / 60, 2)

            results.append({
                'method': method,
                'best_acc': best_acc,
                'training_time': training_time
            })

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(
            os.path.join(self.config.result_dir,
                         f"{self.config.mode}_{self.config.epochs}轮_{time.time()}results.csv"),
            index=False
        )
        return result_df


# 结果分析模块
class ResultAnalyzer:
    @staticmethod
    def visualize_results(config):
        """可视化不同实验的结果"""
        # 读取所有实验结果
        smooth_df = pd.read_csv(os.path.join(config.result_dir, "smooth_results.csv"))
        sharp_df = pd.read_csv(os.path.join(config.result_dir, "sharp_results.csv"))
        combined_df = pd.read_csv(os.path.join(config.result_dir, "combined_results.csv"))

        # 创建对比图表
        plt.figure(figsize=(15, 5))

        # 平滑方法对比
        plt.subplot(1, 3, 1)
        sns.barplot(x='method', y='best_acc', data=smooth_df)
        plt.title('Smoothing Methods Comparison')

        # 锐化方法对比
        plt.subplot(1, 3, 2)
        sns.barplot(x='method', y='best_acc', data=sharp_df)
        plt.title('Sharpening Methods Comparison')

        # 组合效果展示
        plt.subplot(1, 3, 3)
        sns.barplot(x='method', y='best_acc', data=combined_df)
        plt.title('Best Combination Performance')

        plt.tight_layout()
        plt.savefig(os.path.join(config.result_dir, 'comparison.png'))


if __name__ == "__main__":
    # 第一阶段：平滑方法对比
    # smooth_config = ExperimentConfig()
    # smooth_config.mode = "smooth"
    # smooth_runner = ExperimentRunner(smooth_config)
    # smooth_results = smooth_runner.run_experiments()

    # 第二阶段：锐化方法对比
    # sharp_config = ExperimentConfig()
    # sharp_config.mode = "sharp"
    # sharp_runner = ExperimentRunner(sharp_config)
    # sharp_results = sharp_runner.run_experiments()
    # #
    # # 第三阶段：最佳组合验证
    combined_config = ExperimentConfig()
    combined_config.mode = "combined"
    # combined_runner = ExperimentRunner(combined_config)
    # combined_results = combined_runner.run_experiments()

    # 结果可视化
    ResultAnalyzer.visualize_results(combined_config)