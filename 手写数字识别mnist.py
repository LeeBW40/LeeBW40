############################################
# 主要步骤：
#       （1）torch内置函数mnist下载数据。
#       （2）利用torchvision对数据进行预处理，调用torch.utils建立一个数据迭代器。
#       （3）可视化
#       （4）nn。构建神经网络模型
#       （5）模型、损失函数、优化器。
#       （6）训练模型
#       （7）可视化结果
############################################
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn

from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

train_batch_size = 64
test_batch_size = 128
learning_rate = 0.001
num_epochs = 20
momentum = 0.5

transforms = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize([0.5], [0.5])])    # 单通道
#       11、transforms.Compose()方法是将多种变换组合在一起。Compose()会将transforms列表里面的transform操作进行遍历。
#          transforms.ToTensor()  Convert a PIL Image or ndarray to tensor and scale the values accordingly
#       22、torchvision.transforms.Normalize(mean, std)：用给定的均值和标准差分别对每个通道的数据进行正则化。
#           单通道=[0.5], [0.5]     ————     三通道=[m1,m2,m3], [n1,n2,n3]
train_dataset = mnist.MNIST('./pytorch_knowledge', train=True, transform=transforms, download=False)
test_dataset = mnist.MNIST('./pytorch_knowledge', train=False, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

examples = enumerate(test_loader)
print(f"enumerate 的结果是： {examples}")
batch_idx,(example_data, example_label) = next(examples)
print(f"next 结果{next(examples)}")
print(f"example_data  {example_data}")
print(example_data[5]) # torch.Size([128, 1, 28, 28])   (batch_size, channels, height, width)

# batch_idx, (example_data, example_label) = enumerate(test_loader)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none') # 就是该图像的二维数组
    plt.title(f'ground truth{example_label[i]}')
    plt.xticks(([]))
    plt.yticks(([]))
plt.show()

class SimpleNet(nn.Module):
    def __init__(self, input, hidden_1, hidden_2, output):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input, hidden_1), nn.BatchNorm1d(hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(hidden_2, output))

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return x


if __name__ == '__main__':
    # x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    # print(x)
    # print(x.view(x.size(0), -1))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SimpleNet(28*28, 300, 100, 10)
    model.to(device)

    # 损失函数和优化器
    optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate,momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    # print(optimizer.param_groups[0].keys())   #dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach', 'differentiable', 'fused'])
    # train model
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    for epoch in range(num_epochs):
        # 动态修改参数学习率
        if epoch % 5 ==0:
            optimizer.param_groups[0]['lr'] *= 0.1

        # train set =================================================
        train_loss = 0
        train_acc = 0
        model.train()
        for img_data, label_data in train_loader:
            img_data = img_data.to(device)
            label_data = label_data.to(device)
            img_data = img_data.view(img_data.size(0), -1)  # 将高维数据压成 低维数据，按行重组
            # （1）优化器实例化：optimizer
            # （2）前向传播：out = model(img)		# 执行模型中的前向传播得到预测值。
            # （2）损失函数：loss = loss_func(out, label)
            # （3）梯度清零：optimizer.zero_grad()
            # （4）反向传播：loss.backward()     计算梯度
            # （5）参数更新：optimizer.step()     更新参数
            output = model(img_data)                #  前向传播
            loss = criterion(output, label_data)    #  损失函数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            train_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label_data).sum().item()
            ac = num_correct / img_data.shape[0]
            train_acc += ac

        train_loss_temp = train_loss / len(train_loader)
        train_acc_temp = train_acc / len(train_loader)
        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

       # TEST SET
        eval_loss = 0
        eval_acc = 0
        model.eval()
        with torch.no_grad:
            for img_data, label_data in test_loader:
                img_data = img_data.to(device)
                label_data = label_data.to(device)
                img_data = img_data.view(img_data.size(0), -1)  # 将高维数据压成 低维数据，按行重组

                output = model(img_data)  
                loss = criterion(output, label_data)  
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                eval_loss += loss.item()
                _, pred = output.max(1)
                num_correct = (pred == label_data).sum().item()
                ac = num_correct / img_data.shape[0]
                eval_acc += ac

        eval_loss_temp = train_loss / len(train_loader)    # 记录单次测试损失
        eval_acc_temp = train_acc / len(train_loader)      # 记录单次测试准确度
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        print(f"epoch:{epoch}, train_loss:{train_loss_temp:.4f},train_ACC {train_acc_temp:.4f}"
              f"Test_loss:{eval_loss_temp:.4f},test_Acc: {eval_acc_temp:.4f}")

    plt.title('Train LOss')
    plt.plot(np.arange(len(losses)), losses)
    plt.legend(['train loss'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.show()