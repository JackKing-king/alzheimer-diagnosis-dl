"""
数据加载工具
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


class AlzheimerDataset(Dataset):
    """阿尔茨海默病数据集"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Args:
            data_dir: 数据目录路径
            transform: 图像变换
            mode: 'train', 'val', 或 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # 类别映射
        self.class_to_idx = {
            'NonDemented': 0,
            'VeryMildDemented': 1,
            'MildDemented': 2,
            'ModerateDemented': 3
        }
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """加载样本路径和标签"""
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, self.mode, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MRIDataset(Dataset):
    """MRI扫描数据集（支持.nii或.nii.gz格式）"""
    
    def __init__(self, data_csv, transform=None, slice_axis=2):
        """
        Args:
            data_csv: 包含文件路径和标签的CSV文件
            transform: 图像变换
            slice_axis: 切片轴 (0: sagittal, 1: coronal, 2: axial)
        """
        self.data = pd.read_csv(data_csv)
        self.transform = transform
        self.slice_axis = slice_axis
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['file_path']
        label = row['label']
        
        # 加载NIfTI文件
        try:
            import nibabel as nib
            nii_img = nib.load(img_path)
            img_data = nii_img.get_fdata()
            
            # 提取中间切片
            slice_idx = img_data.shape[self.slice_axis] // 2
            if self.slice_axis == 0:
                slice_data = img_data[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_data = img_data[:, slice_idx, :]
            else:
                slice_data = img_data[:, :, slice_idx]
            
            # 归一化到0-255
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            # 转换为PIL图像
            image = Image.fromarray(slice_data).convert('RGB')
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回空白图像
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, augment=True):
    """
    获取数据变换
    
    Args:
        img_size: 图像大小
        augment: 是否使用数据增强
    
    Returns:
        train_transform, val_transform
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    获取数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        img_size: 图像大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_transform = get_transforms(img_size, augment=True)
    
    train_dataset = AlzheimerDataset(data_dir, transform=train_transform, mode='train')
    val_dataset = AlzheimerDataset(data_dir, transform=val_transform, mode='val')
    test_dataset = AlzheimerDataset(data_dir, transform=val_transform, mode='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(dataset):
    """
    计算类别权重（用于处理类别不平衡）
    
    Args:
        dataset: 数据集
    
    Returns:
        weights: 类别权重张量
    """
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    total = len(labels)
    
    # 计算权重：总样本数 / (类别数 * 类别样本数)
    weights = total / (len(class_counts) * class_counts)
    
    return torch.FloatTensor(weights)
