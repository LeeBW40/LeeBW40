import os
import sys
import torch
import json
import torch.nn  as nn
from torchvision import datasets,transforms
import torch.optim as optim
from classic_models.googlenet_v1 import GoogLeNet
from classic_models.alexnet import AlexNet
from tqdm import tqdm

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device")

    data_path = r'D:\Dataset\flower'
    assert os.path.exists(data_path), f"{data_path} doesn't exist."

    # 数据预处理与增强
    """ 
    ToTensor()能够把灰度范围从0-255变换到0-1之间的张量.
    transform.Normalize()则把0-1变换到(-1,1). 具体地说, 对每个通道而言, Normalize执行以下操作: image=(image-mean)/std
    其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1; 而最大值1则变成(1-0.5)/0.5=1. 
    也就是一个均值为0, 方差为1的正态分布. 这样的数据输入格式可以使神经网络更快收敛。
    """
    data_transform = {
        "train": transforms.Compose(
            [transforms.Resize(224),
             transforms.CenterCrop(224),  # 数据增强
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ]
        ),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),  # val不需要数据增强
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 使用imageforder加载数据集中的图像，使用指定的预处理操作来处理图像，返回图像和对应的标签
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=data_transform['val'])
    print(f"train_dataset is {train_dataset}")
    # train_dataset is Dataset ImageFolder
    #     Number of datapoints: 600
    #     Root location: D:\Dataset\flower\train
    #     StandardTransform
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # 使用class_to_idx给类别一个index，作为训练标签
    flower_list = train_dataset.class_to_idx   # class_to_idx (dict): Dict with items (class_name, class_index).
    # 创建一个字典，存储index和类别
    class_dict = dict((value, key) for key, value in flower_list.items())
    json_str = json.dumps(class_dict, indent=4)
    with open(os.path.join(data_path, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)


    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4,shuffle=False)
    print(f"using {train_num} for training,{val_num} images for validation.")


    # 实例化模型
    net = GoogLeNet(num_classes= 5)
    # net = AlexNet(num_classes= 5)

    net.to(device)


    # 指定损失函数，优化器，设置训练迭代论述
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    epochs = 70
    save_path = os.path.abspath(os.path.join(os.curdir, './results/weights/googlenet'))
    # save_path = os.path.abspath(os.path.join(os.curdir, './results/weights/alexnet'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    best_acc = 0.0  # 初始化验证集上最好的准确率，以便后面用该指标筛选模型最优参数
    for epoch in range(epochs):
        net.train()
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

            train_acc = acc_num.item() / sample_num
            train_bar.desc = f"train epoch[{epoch+1}/{epochs}], loss:{loss:.3f}"

        net.eval()
        acc_num = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_images,val_labels = val_data
                outpus = net(val_images.to(device))
                pred_y = torch.max(outpus, dim=1)[1]
                acc_num += torch.eq(pred_y, val_labels.to(device)).sum().item()

        val_acc = acc_num / val_num
        print(f"epoch {epoch+1} train_loss {loss:.3f} train_acc {train_acc:.3f} val_acc {val_acc:.3f}")

        # 判断当前验证集的准确率是否最大，如果是，更新之前保存的权重
        if val_acc>best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(save_path, "GoogleNet.pth"))
            # torch.save(net.state_dict(), os.path.join(save_path, "AlexNet.pth"))


        # 每次迭代后清空指标
        train_acc = 0.0
        val_acc = 0.0

    print('finished training')

main()

