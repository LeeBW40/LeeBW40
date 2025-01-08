import torch.nn as nn
import torch
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        """
        :param num_classes: 分类任务的类别数，1000
        :param init_weights: 是否初始化权重
        """
        super(AlexNet, self).__init__()
        # 特征提取部分，包含多个卷积层和池化层
        # 卷积后图像大小：N = (W-F+2P)/S + 1
        # 池化后图像大小: N = (W-F)/S + 1
        self.features = nn.Sequential(
            # **第一层： ** 输入为224×224×3的图像，经过图像预处理为227×227×3的图像, 卷积核的数量为96，论文中两片GPU分别计算48个核;
            # 卷积核的大小为11×11×3;stride = 4, stride表示的是步长， pad = 0, 表示不扩充边缘。则卷积过后得到96幅图像大小为55×*55
            # 图像，卷积后图像大小计算公式见下。然后进行(LocalResponseNormalized),
            # 最后进行池化操作（MaxPool），pool_size = (3, 3), stride = 2, pad = 0 ，池化后有96幅图像大小为27×27；
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[96, 27, 27]

            nn.Conv2d(96, 256, kernel_size=5, padding=2),           # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 13, 13]

            nn.Conv2d(256, 384, kernel_size=3, padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 6, 6]
        )
        # 分类器部分，包含全连接层和DP层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)                # 输入数据通过特征提取部分
        x = torch.flatten(x, start_dim=1)   # 将特征图展平为一维向量
        x = self.classifier(x)              # 展平后的向量通过分类器部分
        return x

    # 权重初始化方法
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):     # 对于卷积层，使用凯明正态分布初始化权重，并将偏置初始化为0
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):   # 全连接层，使用正态分布初始化权重，将偏置初始化为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(num_classes): 
    model = AlexNet(num_classes=num_classes)
    return model

# net = AlexNet(num_classes=1000)
# summary(net.to('cuda'), (3,224,224))
#########################################################################################################################################
# Total params: 62,378,344
# Trainable params: 62,378,344
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 11.09
# Params size (MB): 237.95
# Estimated Total Size (MB): 249.62
# ----------------------------------------------------------------
# conv_parameters:  3,747,200
# fnn_parameters:  58,631,144   93% 的参数量