import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torch.nn.functional as F


class LiteFaceNet(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()
        # 使用更轻量的MobileNetV2配置
        self.backbone = mobilenet_v2(weights=None, width_mult=0.5).features

        # 修改第一层卷积为1通道输入
        first_conv = self.backbone[0][0]
        self.backbone[0][0] = nn.Conv2d(
            1,  # 输入通道改为1
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        # 冻结更多层以减少训练参数
        for param in self.backbone[:15].parameters():
            param.requires_grad = False

        # 动态确定全连接层输入维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 48, 48)  # 使用实际输入尺寸48x48
            dummy_output = self.backbone(dummy_input)
            self.fc_in_features = dummy_output.shape[1]  # 获取实际通道数

        # 更紧凑的嵌入层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 32),  # 使用动态确定的维度
            nn.ReLU(),
            nn.Linear(32, embedding_size))

        # 量化准备
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2.0, dim=1)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # 为量化融合模型层
        for m in self.modules():
            if type(m) == nn.Sequential:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)