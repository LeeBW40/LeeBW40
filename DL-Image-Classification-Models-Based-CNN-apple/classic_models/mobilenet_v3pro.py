import math
from typing import Callable, List, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

from torchsummary import summary


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,  stride=stride,
                                                         padding=padding, groups=groups, bias=False),
                                                         norm_layer(out_planes),
                                                         activation_layer(inplace=True))

class EfficientChannelAttentionTanh(nn.Module):
    def __init__(self, input_c: int, gamma=2, b=1):
        super().__init__()
        # 动态计算压缩比（参考ECA-Net思想）
        k_size = int(abs((math.log2(input_c) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用1D卷积替代全连接层（减少参数）
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                            padding=(k_size - 1) // 2, bias=False)
        self.tanh = nn.Tanh()  # 核心改进点

    def forward(self, x: Tensor):
        # 通道注意力计算流程
        y = self.avg_pool(x)          # [B,C,1,1]
        y = y.squeeze(-1).transpose(1,2) # [B,1,C]
        y = self.conv(y)              # [B,1,C]
        y = self.tanh(y)              # 输出范围[-1,1]
        y = y.transpose(1,2).unsqueeze(-1) # [B,C,1,1]
        # 特征校准（需进行缩放处理）
        return x * y.expand_as(x) * 0.5 + 0.5  # 将输出映射到[0,1]范围


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=4, kernel_size=3):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力（保持与原始实现相同的激活函数）
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        # 通道注意力
        batch, channel, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(batch, channel))
        max_out = self.mlp(self.max_pool(x).view(batch, channel))
        channel_weights = self.sigmoid(avg_out + max_out).view(batch, channel, 1, 1)
        x = x * channel_weights

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.hardsigmoid(self.conv(spatial))
        return x * spatial_weights


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:
    def __init__(self, input_c: int, kernel: int, expanded_c: int, out_c: int, use_se: bool, activation: str, stride: int,  width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,  cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel, stride=cnf.stride, groups=cnf.expanded_c,
                                        norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))
            # layers.append(EfficientChannelAttentionTanh(cnf.expanded_c))  # 直接替换成改进后的ET网络
        # project
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor):
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class InvertedResidual_CBAM(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual_CBAM, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,  cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel, stride=cnf.stride, groups=cnf.expanded_c,
                                        norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            # layers.append(SqueezeExcitation(cnf.expanded_c))
            # layers.append(EfficientChannelAttentionTanh(cnf.expanded_c))  # 直接替换成改进后的ET网络
            # 或者在特定层使用不同参数
            if cnf.expanded_c > 128:  # 对大通道使用更小的reduction
                layers.append(CBAM(cnf.expanded_c, reduction_ratio=8))
            else:
                layers.append(CBAM(cnf.expanded_c))
        # project
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor):
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result

# 添加CBAM后的Mobile net
class MobileNetV3_CBAM(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int,  num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3_CBAM, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual_CBAM

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3, firstconv_output_c, kernel_size=3, stride=2, norm_layer=norm_layer,  activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c, lastconv_output_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor):
        return self._forward_impl(x)

# 源码：
class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int,  num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3, firstconv_output_c, kernel_size=3, stride=2, norm_layer=norm_layer,  activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c, lastconv_output_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor):
        return self._forward_impl(x)


"""
Constructs a large MobileNetV3 architecture from "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>. 
weights_link:
https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
Args:
    num_classes (int): number of classes
    reduced_tail (bool): If True, reduces the channel counts of all feature layers
        between C4 and C5 by 2. It is used to reduce the channel redundancy in the
        backbone for Detection and Segmentation.
"""
def mobilenet_v3_large(num_classes: int = 1000, reduced_tail: bool = False):

    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1
    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_large_CBAM(num_classes: int = 1000, reduced_tail: bool = False):

    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1
    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3_CBAM(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)

# def mobilenet_v3_large_modified(num_classes=3):
#     model = mobilenet_v3_large(num_classes=num_classes)
#     # 调整最后卷积层通道数（适应小目标）
#     lastconv_input_c = model.features[-2].out_channels
#     model.features[-1] = ConvBNActivation(
#         lastconv_input_c,
#         _make_divisible(512),  # 从1280改为512
#         kernel_size=1,
#         norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
#         activation_layer=nn.Hardswish
#     )
#     # 冻结前3个stage的参数（小数据集时）
#     for param in model.features[:6].parameters():
#         param.requires_grad = False
#     return model



# +++++++++++++++++++将SE替换为优化后的ET后
# net = mobilenet_v3_large(num_classes=1000)
# summary(net.to('cuda'), (3,224,224))
# ================================================================
# Total params: 3,970,786
# Trainable params: 3,970,786
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 109.79
# Params size (MB): 15.15
# Estimated Total Size (MB): 125.51
# ----------------------------------------------------------------
# net=mobilenet_v3_large_CBAM(num_classes=3)
# summary(net.to('cuda'), (3,224,224))

""" 
weights_link:
https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth 
"""
def mobilenet_v3_small(num_classes: int = 1000, reduced_tail: bool = False):

    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


# 测试替换后的模块
def test_cbam_replacement():
    # 原始模型参数
    original_model = mobilenet_v3_large(num_classes=3)
    original_params = sum(p.numel() for p in original_model.parameters())

    # CBAM替换后的模型
    cbam_model = mobilenet_v3_large_CBAM(num_classes=3)
    cbam_params = sum(p.numel() for p in cbam_model.parameters())

    print(f"Original Params: {original_params / 1e6:.2f}M")
    print(f"CBAM Params: {cbam_params / 1e6:.2f}M")   # CBAM Params: 3.46M
    print(f"Params Increase: {(cbam_params - original_params) / 1e6:.2f}M")
    # print(cbam_model)

# 执行测试
test_cbam_replacement()

# Original Params: 4.21M
# CBAM Params: 3.46M
# Params Increase: -0.75M
# cbam_model = mobilenet_v3_large_CBAM(num_classes=3)
# print(cbam_model)
# net = mobilenet_v3_large_CBAM(num_classes=3)
# summary(net.to('cuda'), (3,224,224))