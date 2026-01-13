import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torch.nn.functional as F


class EnhancedFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        # 使用更大的骨干网络
        backbone = mobilenet_v2(weights='DEFAULT').features

        # 解冻更多层进行微调
        for param in backbone.parameters():
            param.requires_grad = True  # 全部解冻

        self.backbone = backbone

        # 增强的分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1280, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return F.normalize(x, p=2, dim=1)