"""
Vision Transformer (ViT)模型
用于阿尔茨海默病MRI图像分类
"""
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """图像分块嵌入"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力到V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 类别token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 分块嵌入
        x = self.patch_embed(x)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 使用类别token进行分类
        cls_output = x[:, 0]
        x = self.head(cls_output)
        
        return x


class ViTWithCNN(nn.Module):
    """ViT与CNN结合的混合模型"""
    
    def __init__(self, img_size=224, patch_size=16, num_classes=4, 
                 embed_dim=768, depth=12, num_heads=12, dropout=0.1):
        super(ViTWithCNN, self).__init__()
        
        import torchvision.models as models
        
        # CNN特征提取器（ResNet18的前几层）
        resnet = models.resnet18(pretrained=True)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:6])
        
        # 计算CNN输出后的特征图大小
        # ResNet前6层将224x224的图像下采样到28x28
        feature_size = img_size // 8  # 224 / 8 = 28
        num_patches = (feature_size // patch_size) ** 2
        
        # 将CNN特征投影到embed_dim
        self.feature_proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        
        # 分块嵌入（对CNN特征图进行分块）
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # 类别token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN特征提取
        x = self.cnn_features(x)
        x = self.feature_proj(x)
        
        # 分块嵌入
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 分类
        cls_output = x[:, 0]
        x = self.head(cls_output)
        
        return x


def get_vit_model(model_name='base', num_classes=4, hybrid=False):
    """
    获取Vision Transformer模型
    
    Args:
        model_name: 模型名称 ('tiny', 'small', 'base', 'large')
        num_classes: 分类数量
        hybrid: 是否使用CNN+Transformer混合架构
    
    Returns:
        model: PyTorch模型
    """
    model_configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")
    
    config = model_configs[model_name]
    
    if hybrid:
        return ViTWithCNN(num_classes=num_classes, **config)
    else:
        return VisionTransformer(num_classes=num_classes, **config)
