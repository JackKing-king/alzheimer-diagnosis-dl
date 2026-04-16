"""
模型对比脚本 - 比较四种深度学习算法
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import get_cnn_model
from models.resnet import get_resnet_model
from models.efficientnet import get_efficientnet_model
from models.vit import get_vit_model
from utils.data_loader import get_data_loaders
from utils.metrics import calculate_metrics


def load_model(model_type, model_name, checkpoint_path, num_classes, device):
    """加载模型"""
    if model_type == 'cnn':
        model = get_cnn_model(model_name, num_classes=num_classes)
    elif model_type == 'resnet':
        model = get_resnet_model(f'resnet{model_name}', num_classes=num_classes)
    elif model_type == 'efficientnet':
        model = get_efficientnet_model(model_name, num_classes=num_classes)
    elif model_type == 'vit':
        model = get_vit_model(model_name, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, test_loader, device):
    """评估模型"""
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


def plot_comparison(results, save_path):
    """绘制模型对比图"""
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        values = [results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_heatmap(results, save_path):
    """绘制指标热力图"""
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    
    data = []
    for model in models:
        row = [results[model][metric] for metric in metrics_names]
        data.append(row)
    
    df = pd.DataFrame(data, index=models, columns=metrics_names)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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
    
    # 模型配置
    models_config = {
        'CNN': {
            'type': 'cnn',
            'name': 'resnet50',
            'checkpoint': args.cnn_checkpoint
        },
        'ResNet': {
            'type': 'resnet',
            'name': '50',
            'checkpoint': args.resnet_checkpoint
        },
        'EfficientNet': {
            'type': 'efficientnet',
            'name': 'b0',
            'checkpoint': args.efficientnet_checkpoint
        },
        'ViT': {
            'type': 'vit',
            'name': 'base',
            'checkpoint': args.vit_checkpoint
        }
    }
    
    results = {}
    
    # 评估每个模型
    for model_name, config in models_config.items():
        if config['checkpoint'] and os.path.exists(config['checkpoint']):
            print(f"\nEvaluating {model_name}...")
            
            model = load_model(
                config['type'],
                config['name'],
                config['checkpoint'],
                args.num_classes,
                device
            )
            
            y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
            metrics = calculate_metrics(y_true, y_pred, num_classes=args.num_classes)
            
            results[model_name] = metrics
            
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            print(f"Skipping {model_name} - checkpoint not found")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 创建对比表格
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(args.output_dir, 'comparison_results.csv'))
    
    # 绘制对比图
    plot_comparison(results, os.path.join(args.output_dir, 'model_comparison.png'))
    plot_metrics_heatmap(results, os.path.join(args.output_dir, 'metrics_heatmap.png'))
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    
    # 找出最佳模型
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\nBest Model (by Accuracy): {best_model}")
    print(f"  Accuracy:  {results[best_model]['accuracy']:.4f}")
    print(f"  Precision: {results[best_model]['precision']:.4f}")
    print(f"  Recall:    {results[best_model]['recall']:.4f}")
    print(f"  F1-Score:  {results[best_model]['f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare all models')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='results/comparison')
    parser.add_argument('--cnn_checkpoint', type=str, default='results/cnn/checkpoints/best_model.pth')
    parser.add_argument('--resnet_checkpoint', type=str, default='results/resnet/checkpoints/best_model.pth')
    parser.add_argument('--efficientnet_checkpoint', type=str, default='results/efficientnet/checkpoints/best_model.pth')
    parser.add_argument('--vit_checkpoint', type=str, default='results/vit/checkpoints/best_model.pth')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
