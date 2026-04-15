# 模型下载脚本
# 用于下载预训练模型和数据集

import os
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """下载文件"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_alzheimer_dataset():
    """
    下载阿尔茨海默病数据集
    推荐使用Kaggle的Alzheimer MRI Preprocessed Dataset
    """
    print("=" * 60)
    print("阿尔茨海默病数据集下载指南")
    print("=" * 60)
    print("""
推荐使用以下公开数据集：

1. Kaggle - Alzheimer MRI Preprocessed Dataset
   链接: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-preprocessed-dataset
   包含4个类别：
   - NonDemented (正常)
   - VeryMildDemented (非常轻度痴呆)
   - MildDemented (轻度痴呆)
   - ModerateDemented (中度痴呆)

2. OASIS Brains Dataset
   链接: https://www.oasis-brains.org/
   需要注册申请

3. ADNI (Alzheimer's Disease Neuroimaging Initiative)
   链接: https://adni.loni.usc.edu/
   需要注册申请

下载后请将数据组织为以下结构：
data/
├── train/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
├── val/
│   └── ...
└── test/
    └── ...

或者运行预处理脚本：
python utils/preprocessing.py --task organize --source_dir <原始数据路径> --output_dir data
""")


def setup_project():
    """设置项目环境"""
    print("=" * 60)
    print("项目设置")
    print("=" * 60)
    
    # 创建必要的目录
    dirs = [
        'data/raw', 'data/processed', 'data/splits',
        'results/cnn/checkpoints', 'results/cnn/logs',
        'results/resnet/checkpoints', 'results/resnet/logs',
        'results/efficientnet/checkpoints', 'results/efficientnet/logs',
        'results/vit/checkpoints', 'results/vit/logs',
        'results/figures'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")
    
    print("\n项目设置完成！")
    print("\n下一步：")
    print("1. 下载数据集（见上面的指南）")
    print("2. 安装依赖: pip install -r requirements.txt")
    print("3. 开始训练: python training/train_cnn.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', action='store_true', help='Setup project directories')
    parser.add_argument('--dataset', action='store_true', help='Show dataset download guide')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_project()
    elif args.dataset:
        download_alzheimer_dataset()
    else:
        print("请使用 --setup 设置项目或使用 --dataset 查看数据集下载指南")
