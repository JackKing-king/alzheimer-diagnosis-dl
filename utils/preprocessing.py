"""
数据预处理脚本
"""
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
from tqdm import tqdm


def organize_dataset(source_dir, output_dir, val_split=0.15, test_split=0.15):
    """
    组织数据集为训练/验证/测试集
    
    Args:
        source_dir: 源数据目录（包含类别子目录）
        output_dir: 输出目录
        val_split: 验证集比例
        test_split: 测试集比例
    """
    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # 遍历每个类别
    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        print(f"Processing class: {class_name}")
        
        # 获取该类别所有图像
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            continue
        
        # 划分数据集
        train_imgs, temp_imgs = train_test_split(
            images, test_size=val_split + test_split, random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, 
            test_size=test_split / (val_split + test_split), 
            random_state=42
        )
        
        # 复制文件到相应目录
        for split, imgs in zip(splits, [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img_name in tqdm(imgs, desc=f'{split}/{class_name}'):
                src_path = os.path.join(class_dir, img_name)
                dst_path = os.path.join(split_class_dir, img_name)
                shutil.copy2(src_path, dst_path)
    
    print(f"\nDataset organized in: {output_dir}")
    
    # 打印统计信息
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            total_images = sum([
                len([f for f in os.listdir(os.path.join(split_dir, d)) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ])
            print(f"{split}: {total_images} images")


def create_metadata(data_dir, output_file):
    """
    创建数据集元数据CSV文件
    
    Args:
        data_dir: 数据目录
        output_file: 输出CSV文件路径
    """
    data = []
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    data.append({
                        'file_path': img_path,
                        'file_name': img_name,
                        'class': class_name,
                        'split': split
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Metadata saved to: {output_file}")
    print(f"Total samples: {len(df)}")
    print(df['split'].value_counts())


def resize_images(source_dir, output_dir, target_size=(224, 224)):
    """
    调整图像大小
    
    Args:
        source_dir: 源数据目录
        output_dir: 输出目录
        target_size: 目标大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in tqdm(os.listdir(source_dir), desc='Resizing'):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                output_path = os.path.join(output_class_dir, img_name)
                
                try:
                    img = Image.open(img_path)
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    print(f"Resized images saved to: {output_dir}")


def main(args):
    if args.task == 'organize':
        organize_dataset(args.source_dir, args.output_dir, args.val_split, args.test_split)
    elif args.task == 'metadata':
        create_metadata(args.data_dir, args.output_file)
    elif args.task == 'resize':
        resize_images(args.source_dir, args.output_dir, tuple(args.target_size))
    else:
        print(f"Unknown task: {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing utilities')
    parser.add_argument('--task', type=str, required=True,
                       choices=['organize', 'metadata', 'resize'],
                       help='Task to perform')
    parser.add_argument('--source_dir', type=str, help='Source directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--output_file', type=str, help='Output file path')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224])
    
    args = parser.parse_args()
    main(args)
