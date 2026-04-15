"""
评估脚本 - 评估单个模型
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import get_cnn_model
from models.resnet import get_resnet_model
from models.efficientnet import get_efficientnet_model
from models.vit import get_vit_model
from utils.data_loader import get_data_loaders
from utils.metrics import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve,
    print_classification_report
)


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    _, _, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # 类别名称
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    # 根据模型类型创建模型
    print(f"Loading {args.model_type} model...")
    if args.model_type == 'cnn':
        model = get_cnn_model(args.model_name, num_classes=args.num_classes)
    elif args.model_type == 'resnet':
        model = get_resnet_model(f'resnet{args.model_name}', num_classes=args.num_classes)
    elif args.model_type == 'efficientnet':
        model = get_efficientnet_model(args.model_name, num_classes=args.num_classes)
    elif args.model_type == 'vit':
        model = get_vit_model(args.model_name, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 评估
    print("\nEvaluating model...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, num_classes=args.num_classes)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    # 打印分类报告
    print_classification_report(y_true, y_pred, class_names)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存指标
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    print(f"\nConfusion matrix saved to {args.output_dir}/confusion_matrix.png")
    
    # 绘制ROC曲线
    plot_roc_curve(
        y_true, y_probs, num_classes=args.num_classes,
        class_names=class_names,
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )
    print(f"ROC curve saved to {args.output_dir}/roc_curve.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/evaluation')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['cnn', 'resnet', 'efficientnet', 'vit'])
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
