# 基于深度学习的阿尔茨海默病诊断项目

## 项目概述
本项目使用四种深度学习算法对阿尔茨海默病（Alzheimer's Disease, AD）进行诊断分类。

## 数据集
使用公开数据集：
- **ADNI (Alzheimer's Disease Neuroimaging Initiative)** - 神经影像数据
- **OASIS (Open Access Series of Imaging Studies)** - 脑部MRI扫描数据
- **Kaggle Alzheimer's Dataset** - 预处理好的图像分类数据集

## 四种深度学习算法

### 1. CNN (卷积神经网络)
- 基础架构：ResNet50 / VGG16
- 用途：直接从MRI图像中提取特征进行分类

### 2. ResNet (残差网络)
- 架构：ResNet101 / ResNet152
- 优势：解决深层网络梯度消失问题，适合医学图像

### 3. EfficientNet
- 架构：EfficientNet-B0 到 B7
- 优势：在准确率和计算效率之间取得平衡

### 4. Vision Transformer (ViT)
- 架构：ViT-Base / ViT-Large
- 优势：利用自注意力机制捕获全局特征

## 项目结构
```
基于深度学习的阿尔茨海默病诊断/
├── data/                   # 数据集目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理后的数据
│   └── splits/            # 训练/验证/测试集划分
├── models/                # 模型定义
│   ├── cnn.py
│   ├── resnet.py
│   ├── efficientnet.py
│   └── vit.py
├── training/              # 训练脚本
│   ├── train_cnn.py
│   ├── train_resnet.py
│   ├── train_efficientnet.py
│   └── train_vit.py
├── evaluation/            # 评估脚本
│   ├── evaluate.py
│   └── compare_models.py
├── utils/                 # 工具函数
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── metrics.py
├── results/               # 实验结果
│   ├── checkpoints/       # 模型检查点
│   ├── logs/             # 训练日志
│   └── figures/          # 可视化图表
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Python依赖
└── README.md             # 项目说明

```

## 分类任务
- **二分类**：正常 (NC) vs 阿尔茨海默病 (AD)
- **三分类**：正常 (NC) vs 轻度认知障碍 (MCI) vs 阿尔茨海默病 (AD)
- **四分类**：正常 (CN) vs 轻度认知障碍早期 (EMCI) vs 轻度认知障碍晚期 (LMCI) vs 阿尔茨海默病 (AD)

## 环境要求
- Python 3.8+
- PyTorch 1.10+
- torchvision
- timm (用于Vision Transformer)
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 下载数据集并放入 `data/raw/` 目录
3. 运行数据预处理：`python utils/preprocessing.py`
4. 训练模型：`python training/train_cnn.py`
5. 评估模型：`python evaluation/evaluate.py`

## 评估指标
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1-Score
- AUC-ROC
- 混淆矩阵

## GitHub Actions 自动训练

本项目配置了 GitHub Actions 自动化工作流，支持云端自动训练：

### 触发方式

1. **手动触发**: 进入仓库的 Actions 页面，选择 workflow 后点击 "Run workflow"
2. **定时触发**: 每周日凌晨 2 点自动运行

### 可用的 Workflows

#### 1. Train Alzheimer Diagnosis Models
训练深度学习模型，支持：
- **CNN** (ResNet50)
- **ResNet** (ResNet101)
- **EfficientNet** (EfficientNet-B0)
- **ViT** (Vision Transformer)
- **All** (同时训练所有模型)

参数配置：
- 训练轮数 (epochs)
- 批次大小 (batch_size)

训练完成后自动上传：
- 模型检查点 (checkpoints)
- 训练日志 (logs)
- 训练历史 (history.json)

#### 2. Data Preparation
数据预处理 workflow：
- 下载数据集
- 数据分割 (train/val/test)
- 数据缓存

#### 3. Model Evaluation
模型评估 workflow：
- 加载训练好的模型
- 在测试集上评估
- 生成评估报告

### 使用步骤

1. 进入仓库的 **Actions** 标签页
2. 选择要运行的 workflow
3. 点击 **Run workflow**
4. 选择模型类型和参数
5. 点击 **Run workflow** 开始训练
6. 训练完成后在 **Artifacts** 中下载结果

### 注意事项

- GitHub Actions 免费版有 2000 分钟/月的限制
- 训练使用 CPU 运行，速度较慢
- 建议上传预处理好的数据到 GitHub Release 或外部存储

## 参考文献
1. Alzheimer's Disease Neuroimaging Initiative (ADNI)
2. OASIS Brains Project
3. He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
4. Tan, M., & Le, Q. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.
5. Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021.
