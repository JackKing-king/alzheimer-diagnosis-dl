"""
CNN模型 - 卷积神经网络
用于阿尔茨海默病MRI图像分类
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNModel(nn.Module):
    """基于ResNet50的CNN模型"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(CNNModel, self).__init__()
        
        # 加载预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 修改第一层以适应灰度图像（MRI通常是单通道）
        # 或者保持3通道，将灰度图复制到3个通道
        in_features = self.backbone.fc.in_features
        
        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class VGGModel(nn.Module):
    """基于VGG16的CNN模型"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(VGGModel, self).__init__()
        
        # 加载预训练的VGG16
        self.backbone = models.vgg16(pretrained=pretrained)
        
        # 替换分类器
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SimpleCNN(nn.Module):
    """简单的自定义CNN架构"""
    
    def __init__(self, num_classes=4, input_channels=3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 卷积块1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # 卷积块2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # 卷积块3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # 卷积块4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_cnn_model(model_name='resnet50', num_classes=4, pretrained=True):
    """
    获取CNN模型
    
    Args:
        model_name: 模型名称 ('resnet50', 'vgg16', 'simple')
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: PyTorch模型
    """
    if model_name == 'resnet50':
        return CNNModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vgg16':
        return VGGModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'simple':
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
