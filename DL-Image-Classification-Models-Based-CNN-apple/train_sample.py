import os
import sys

import locale

# 设置系统默认编码
if sys.platform == 'win32':
    locale.setlocale(locale.LC_ALL, 'chinese')  # Windows简中环境

import cv2
import torch
import json
import time

import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,LambdaLR
from datetime import datetime
from classic_models.googlenet_v1 import GoogLeNet
from classic_models.alexnet import AlexNet
from classic_models.mobilenet_v3 import MobileNetV3,mobilenet_v3_large
from classic_models.mobilenet_v3pro import mobilenet_v3_large_CBAM,MobileNetV3_CBAM  # "mobilenetV3" + "mobilenetV3+CBAM" + "mobileNetV3+Ghost"
from classic_models.mobilenet_v3_ghost import mobilenet_v3_large_ghost
from classic_models.resnet import ResNet,resnet101
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.visualize_GradCAM import visualize_gradcam


# 转换适配器（解决PIL.Image到numpy的转换）
class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        img_np = np.array(img)              # PIL转numpy
        augmented = self.aug(image=img_np)
        return augmented['image']


def warmup_scheduler(step, warmup_steps, initial_lr):
    if step < warmup_steps:
        return (step + 1) / warmup_steps * initial_lr
    return initial_lr


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device")
    timestamp = time.time()
    data_path = r'D:\post-file\DL-Image-Classification-Models-Based-CNN\dataset_unit\apple'    # 改成苹果数据集（1，2，3）
    assert os.path.exists(data_path), f"{data_path} doesn't exist."

    # 数据预处理与增强
    aug = A.Compose([
        A.Resize(height=224, width=224),  # 统一尺寸处理
        A.Rotate(limit=(-15, 15), p=0.6),   # 旋转
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=0
        ),
        A.RandomResizedCrop(
            size=(224,224),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        # A.GaussianBlur(
        #     blur_limit=(3, 7),
        #     sigma_limit=0.5,
        #     p=0.5
        # ),
        A.Normalize(  # 归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()  # 转换到Tensor
    ])

    """ 
    ToTensor()能够把灰度范围从0-255变换到0-1之间的张量.
    transform.Normalize()则把0-1变换到(-1,1). 具体地说, 对每个通道而言, Normalize执行以下操作: image=(image-mean)/std
    其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1; 而最大值1则变成(1-0.5)/0.5=1. 
    也就是一个均值为0, 方差为1的正态分布. 这样的数据输入格式可以使神经网络更快收敛。
    """
    data_transform = {
        # "train": transforms.Compose(
        #     [transforms.Resize(224),
        #      transforms.CenterCrop(224),  # 数据增强
        #      transforms.ToTensor(),
        #      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准归一化
        #      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #      ]
        # ),
        "train": transforms.Compose([
             AlbumentationsTransform(aug),
            ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),  # val不需要数据增强
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准归一化
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 使用imageforder加载数据集中的图像，使用指定的预处理操作来处理图像，返回图像和对应的标签
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=data_transform['val'])
    print(f"train_dataset is {train_dataset}")
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # 使用class_to_idx给类别一个index，作为训练标签
    apple_list = train_dataset.class_to_idx            # class_to_idx (dict): Dict with items (class_name, class_index).
    # 创建一个字典，存储index和类别
    class_dict = dict((value, key) for key, value in apple_list.items())
    json_str = json.dumps(class_dict, indent=4)
    with open(os.path.join(data_path, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
    print(f"using {train_num} for training,{val_num} images for validation.")

    # 实例化模型
    # net = GoogLeNet(num_classes=3)
    # net = mobilenet_v3_large_CBAM(num_classes=3)
    # net = mobilenet_v3_large_ghost(num_classes=3)
    # net = mobilenet_v3_large(num_classes=3)
    net = AlexNet(num_classes= 3)
    # net = resnet101(num_classes=3)
    net.to(device)

    # 指定损失函数，优化器，设置训练迭代论述
    lr = 0.0002
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #  STepLr
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # scheduler = LambdaLR(optimizer,lr_lambda=lambda step:warmup_scheduler(step, warmup_steps=100, initial_lr=0.1))
    # optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)    # sgd训练精度比adam低
    epochs = 5
    weights_list = ['GoogleNet', 'MobileNetV3', 'AlexNet', "ResNet","MobileNetV3_CBAM","MobileNetV3_Ghost"]
    save_path = os.path.abspath(os.path.join(os.curdir, f'./results/weights/{weights_list[2]}'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result_dir = os.path.abspath(os.path.join(os.curdir, './results/plots'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0  # 新增：记录每个epoch的总损失
        acc_num = torch.zeros(1).to(device)
        sample_num = 0    # 用于记录当前迭代中，已经计算量多少个样本

        train_bar = tqdm(train_loader, file=sys.stdout,ncols=100)
        for data in train_bar:
            images,labels = data
            sample_num += images.shape[0]
            print(images.shape)

            optimizer.zero_grad()
            outputs = net(images.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(pred_class, labels.to(device)).sum()

            loss = loss_func(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # scheduler.step()  # 更新学习率

            running_loss += loss.item()  # 新增：累计损失值
            # train_acc = acc_num.item() / sample_num
            train_bar.desc = f"train epoch[{epoch+1}/{epochs}], loss:{loss:.3f}"

        # 计算平均训练损失
        epoch_train_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_train_loss)

        # 计算训练准确率
        epoch_train_acc = acc_num.item() / sample_num
        train_acc_list.append(epoch_train_acc)

        net.eval()
        acc_num = 0.0
        val_acc_num = 0.0
        val_running_loss = 0.0  # 新增，验证机损失

        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outpus = net(val_images.to(device))

                val_loss = loss_func(outpus, val_labels.to(device))
                val_running_loss += val_loss.item()

                pred_y = torch.max(outpus, dim=1)[1]
                acc_num += torch.eq(pred_y, val_labels.to(device)).sum().item()
                val_acc_num += torch.eq(pred_y, val_labels.to(device)).sum().item()

        # 计算验证指标
        epoch_val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(epoch_val_loss)

        epoch_val_acc = val_acc_num / val_num
        val_acc_list.append(epoch_val_acc)

        # val_acc = acc_num / val_num
        # print(f"epoch {epoch+1} train_loss {loss:.3f} train_acc {train_acc:.3f} ----"
        #       f"----- val_loss{val_loss:.3f} val_acc {val_acc:.3f}")

        print(f"epoch {epoch + 1}: "
              f"train_loss {epoch_train_loss:.3f} | "
              f"train_acc {epoch_train_acc:.3f} | "
              f"val_loss {epoch_val_loss:.3f} | "
              f"val_acc {epoch_val_acc:.3f}")

        # 判断当前验证集的准确率是否最大，如果是，更新之前保存的权重
        # if val_acc >best_acc:
        #     best_acc = val_acc
        #     torch.save(net.state_dict(), os.path.join(save_path, f"{weights_list[0]}.pth"))

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(net.state_dict(), os.path.join(save_path, f"{weights_list[2]}.pth"))

        # 每次迭代后清空指标
        # train_acc = 0.0
        # val_acc = 0.0
        epoch_train_acc = 0.0
        epoch_val_acc = 0.0

    # print('finished training')
    end_time = time.time()
    training_time = end_time - start_time
    per_time = training_time/epochs
    print(f"{weights_list[2]}: training_time {training_time:.2f} (Batch Size={batch_size})")  #
    print(f"{weights_list[2]}: {per_time:.2f} seconds per epoch (Batch Size={batch_size})")  # 计算速度

    plt.figure(figsize=(12, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像
    plt.savefig(os.path.join(result_dir, f'training{epochs}轮_lr{lr}_batchsize_{batch_size}_metrics_{weights_list[2]}_{timestamp}.png'))
    plt.close()

    print('Finished Training')

    # 加载最佳模型进行可视化
    best_model = AlexNet(num_classes=3)
    best_model.load_state_dict(
        torch.load(
            os.path.join(save_path, f"{weights_list[2]}.pth"),
            map_location=device,
            weights_only=True  # 安全限制
        )
    )
    # best_model.load_state_dict(torch.load(os.path.join(save_path, f"{weights_list[2]}.pth")))
    best_model.to(device).eval()

    # run Grad-CAM
    cam_save_dir = os.path.abspath(os.path.join(os.curdir, f'results/gradcam/gradcam_{weights_list[2]}'))
    visualize_gradcam(best_model, device, data_path, cam_save_dir)


if __name__ == '__main__':
    main()

