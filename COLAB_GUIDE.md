# Google Colab 训练指南

## 快速开始

点击下面的按钮直接在 Google Colab 中打开项目：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackKing-king/alzheimer-diagnosis-dl/blob/main/colab_train.ipynb)

## 使用步骤

### 1. 准备数据

#### 方法 A: 使用 Google Drive（推荐）
1. 将预处理好的数据上传到 Google Drive:
   - 路径: `MyDrive/alzheimer-data/`
   - 结构:
     ```
     alzheimer-data/
     ├── splits/
     │   ├── train/
     │   ├── val/
     │   └── test/
     ```

2. 在 Colab 中运行时会自动挂载 Drive 并加载数据

#### 方法 B: 从 Kaggle 下载
1. 获取 Kaggle API Token:
   - 访问 https://www.kaggle.com/account
   - 点击 "Create New API Token"
   - 下载 `kaggle.json`

2. 在 Colab 中上传 `kaggle.json` 文件

3. 取消 notebook 中相关代码的注释来下载数据

#### 方法 C: 使用 Google Drive 分享链接
1. 将数据压缩上传到 Google Drive
2. 获取分享链接的 FILE_ID
3. 修改 notebook 中的 FILE_ID 变量

### 2. 运行训练

1. 打开 Colab Notebook
2. 点击 **Runtime** → **Change runtime type**
3. 选择 **GPU** 作为硬件加速器
4. 依次运行所有代码单元格
5. 或使用 **Runtime** → **Run all**

### 3. 配置参数

在 notebook 中修改以下变量：

```python
MODEL = "cnn"        # 可选: cnn, resnet, efficientnet, vit
EPOCHS = 50          # 训练轮数
BATCH_SIZE = 32      # 批次大小
LEARNING_RATE = 1e-4 # 学习率
```

### 4. 查看结果

训练完成后，结果会自动保存到 Google Drive:
- 模型检查点: `MyDrive/alzheimer-results/{MODEL}/checkpoints/`
- 训练日志: `MyDrive/alzheimer-results/{MODEL}/logs/`
- 评估结果: `MyDrive/alzheimer-results/{MODEL}/evaluation/`
- 训练曲线: `MyDrive/alzheimer-results/{MODEL}/training_curves.png`

## 训练多个模型

如果要训练所有模型并对比，可以：

1. 依次运行不同模型的训练单元格
2. 或者使用以下代码训练所有模型:

```python
models = ["cnn", "resnet", "efficientnet", "vit"]
for model in models:
    MODEL = model
    # 运行训练代码
```

## 注意事项

- Colab 免费版 GPU 有时长限制（约 12 小时）
- 建议定期保存检查点到 Google Drive
- 如果断开连接，可以从检查点恢复训练

## 数据下载链接

### Kaggle 数据集
- 数据集: [Alzheimer's Dataset (4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- 包含: MildDemented, ModerateDemented, NonDemented, VeryMildDemented

### 数据预处理
如果下载的是原始数据，运行以下命令进行预处理:
```python
!python utils/preprocessing.py --task organize --source_dir data/raw --output_dir data/splits
```

## 故障排除

### GPU 不可用
```python
# 检查 CUDA
import torch
print(torch.cuda.is_available())
```
如果返回 False，请检查 runtime 类型是否设置为 GPU

### 内存不足
- 减小 BATCH_SIZE
- 使用更小的模型（如 efficientnet_b0 代替 efficientnet_b4）

### 断开连接
- 使用 `drive.mount()` 确保数据持久化
- 定期保存模型检查点
