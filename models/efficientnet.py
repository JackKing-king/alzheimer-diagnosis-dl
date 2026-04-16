"""
EfficientNet模型
用于阿尔茨海默病MRI图像分类
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetModel(nn.Module):
    """EfficientNet模型基类"""
    
    def __init__(self, backbone, num_classes=4, dropout=0.5):
        super(EfficientNetModel, self).__init__()
        
        self.backbone = backbone
        
        # 获取最后一层的输入特征数
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                # 替换分类器
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(dropout),
                    nn.Linear(512, num_classes)
                )
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, num_classes)
                )
    
    def forward(self, x):
        return self.backbone(x)


def get_efficientnet_b0(num_classes=4, pretrained=True, dropout=0.5):
    """获取EfficientNet-B0模型"""
    try:
        backbone = models.efficientnet_b0(pretrained=pretrained)
    except:
        # 如果torchvision版本较旧，使用weights参数
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    return EfficientNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_efficientnet_b1(num_classes=4, pretrained=True, dropout=0.5):
    """获取EfficientNet-B1模型"""
    try:
        backbone = models.efficientnet_b1(pretrained=pretrained)
    except:
        backbone = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
    return EfficientNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_efficientnet_b2(num_classes=4, pretrained=True, dropout=0.5):
    """获取EfficientNet-B2模型"""
    try:
        backbone = models.efficientnet_b2(pretrained=pretrained)
    except:
        backbone = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
    return EfficientNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_efficientnet_b3(num_classes=4, pretrained=True, dropout=0.5):
    """获取EfficientNet-B3模型"""
    try:
        backbone = models.efficientnet_b3(pretrained=pretrained)
    except:
        backbone = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
    return EfficientNetModel(backbone, num_classes=num_classes, dropout=dropout)


def get_efficientnet_b4(num_classes=4, pretrained=True, dropout=0.5):
    """获取EfficientNet-B4模型"""
    try:
        backbone = models.efficientnet_b4(pretrained=pretrained)
    except:
        backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
    return EfficientNetModel(backbone, num_classes=num_classes, dropout=dropout)


class EfficientNetWithFeaturePyramid(nn.Module):
    """带特征金字塔的EfficientNet"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(EfficientNetWithFeaturePyramid, self).__init__()
        
        # 加载EfficientNet-B0
        try:
            efficientnet = models.efficientnet_b0(pretrained=pretrained)
        except:
            efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        
        # 提取特征层
        self.features = efficientnet.features
        
        # 特征金字塔层
        self.fpn_conv1 = nn.Conv2d(320, 256, kernel_size=1)  # 最高层特征
        self.fpn_conv2 = nn.Conv2d(112, 256, kernel_size=1)
        self.fpn_conv3 = nn.Conv2d(40, 256, kernel_size=1)
        self.fpn_conv4 = nn.Conv2d(24, 256, kernel_size=1)
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 提取多尺度特征
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # 保存特定层的输出
            if i in [2, 4, 6, 8]:  # 对应不同分辨率
                features.append(x)
        
        # 特征金字塔融合
        fpn_features = []
        fpn_features.append(self.fpn_conv1(features[-1]))
        
        for i in range(len(features) - 2, -1, -1):
            fpn_conv = [self.fpn_conv4, self.fpn_conv3, self.fpn_conv2][i]
            upsampled = self.upsample(fpn_features[-1])
            fpn_features.append(fpn_conv(features[i]) + upsampled)
        
        # 全局平均池化
        pooled_features = []
        for f in fpn_features:
            pooled = nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
            pooled_features.append(pooled)
        
        # 拼接所有特征
        x = torch.cat(pooled_features, dim=1)
        
        # 分类
        x = self.classifier(x)
        return x


class EfficientNetV2Model(nn.Module):
    """EfficientNet-V2模型"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(EfficientNetV2Model, self).__init__()
        
        try:
            # 尝试加载EfficientNet-V2
            self.backbone = models.efficientnet_v2_s(pretrained=pretrained)
        except:
            # 如果不可用，回退到B0
            print("EfficientNet-V2 not available, falling back to EfficientNet-B0")
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # 修改分类器
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(dropout),
                    nn.Linear(512, num_classes)
                )
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, num_classes)
                )
    
    def forward(self, x):
        return self.backbone(x)


def get_efficientnet_model(model_name='b0', num_classes=4, pretrained=True, fpn=False):
    """
    获取EfficientNet模型
    
    Args:
        model_name: 模型名称 ('b0', 'b1', 'b2', 'b3', 'b4', 'v2-s')
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        fpn: 是否使用特征金字塔
    
    Returns:
        model: PyTorch模型
    """
    if fpn:
        return EfficientNetWithFeaturePyramid(num_classes=num_classes, pretrained=pretrained)
    
    if model_name == 'v2-s':
        return EfficientNetV2Model(num_classes=num_classes, pretrained=pretrained)
    
    model_map = {
        'b0': get_efficientnet_b0,
        'b1': get_efficientnet_b1,
        'b2': get_efficientnet_b2,
        'b3': get_efficientnet_b3,
        'b4': get_efficientnet_b4
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model_map[model_name](num_classes=num_classes, pretrained=pretrained)
