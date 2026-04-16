"""
ResNet模型 - 残差网络
用于阿尔茨海默病MRI图像分类
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    """ResNet模型基类"""
    
    def __init__(self, backbone, num_classes=4, dropout=0.5):
        super(ResNetModel, self).__init__()
        
        self.backbone = backbone
        in_features = self.backbone.fc.in_features
        
        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_resnet18(num_classes=4, pretrained=True, dropout=0.5):
    """获取ResNet18模型"""
    backbone = models.resnet18(pretrained=pretrained)
    return ResNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_resnet34(num_classes=4, pretrained=True, dropout=0.5):
    """获取ResNet34模型"""
    backbone = models.resnet34(pretrained=pretrained)
    return ResNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_resnet50(num_classes=4, pretrained=True, dropout=0.5):
    """获取ResNet50模型"""
    backbone = models.resnet50(pretrained=pretrained)
    return ResNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_resnet101(num_classes=4, pretrained=True, dropout=0.5):
    """获取ResNet101模型"""
    backbone = models.resnet101(pretrained=pretrained)
    return ResNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_resnet152(num_classes=4, pretrained=True, dropout=0.5):
    """获取ResNet152模型"""
    backbone = models.resnet152(pretrained=pretrained)
    return ResNetModel(backbone, num_classes=num_classes, dropout=dropout)


class ResNetWithAttention(nn.Module):
    """带注意力机制的ResNet"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(ResNetWithAttention, self).__init__()
        
        # 加载ResNet50并移除最后的全连接层
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        x = self.features(x)
        
        # 应用空间注意力
        spatial_attn = self.spatial_attention(x)
        x = x * spatial_attn
        
        # 应用通道注意力
        channel_attn = self.channel_attention(x)
        channel_attn = channel_attn.view(channel_attn.size(0), 2048, 1, 1)
        x = x * channel_attn
        
        # 分类
        x = self.classifier(x)
        return x


class MultiScaleResNet(nn.Module):
    """多尺度ResNet - 融合不同尺度的特征"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(MultiScaleResNet, self).__init__()
        
        # 加载ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # 提取不同层的特征
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 56x56
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # 28x28
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # 14x14
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])  # 7x7
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048 + 1024 + 512 + 256, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 提取多尺度特征
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # 全局平均池化并展平
        x1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1)).flatten(1)
        x2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1)).flatten(1)
        x3 = nn.functional.adaptive_avg_pool2d(x3, (1, 1)).flatten(1)
        x4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1)).flatten(1)
        
        # 拼接特征
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        # 融合分类
        x = self.fusion(x)
        return x


def get_resnet_model(model_name='resnet50', num_classes=4, pretrained=True, attention=False, multiscale=False):
    """
    获取ResNet模型
    
    Args:
        model_name: 模型名称 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        attention: 是否使用注意力机制
        multiscale: 是否使用多尺度融合
    
    Returns:
        model: PyTorch模型
    """
    if attention:
        return ResNetWithAttention(num_classes=num_classes, pretrained=pretrained)
    
    if multiscale:
        return MultiScaleResNet(num_classes=num_classes, pretrained=pretrained)
    
    model_map = {
        'resnet18': get_resnet18,
        'resnet34': get_resnet34,
        'resnet50': get_resnet50,
        'resnet101': get_resnet101,
        'resnet152': get_resnet152
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model_map[model_name](num_classes=num_classes, pretrained=pretrained)
