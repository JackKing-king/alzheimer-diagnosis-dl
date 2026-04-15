"""
数据上传脚本 - 用于将本地数据上传到 Google Drive 以便在 Colab 中使用
"""
import os
import shutil
from pathlib import Path

def prepare_data_for_upload():
    """
    准备数据上传到 Google Drive
    这会创建一个压缩包，方便上传到 Google Drive
    """
    
    # 检查数据是否存在
    data_dir = Path("data/splits")
    if not data_dir.exists():
        print("错误: 数据目录不存在!")
        print("请先运行: python utils/preprocessing.py --task organize --source_dir data/raw --output_dir data/splits")
        return
    
    # 统计数据
    print("=" * 60)
    print("数据统计:")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            class_counts = {}
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    class_counts[class_dir.name] = count
            
            total = sum(class_counts.values())
            print(f"\n{split.upper()}:")
            for cls, count in sorted(class_counts.items()):
                print(f"  {cls}: {count} 张")
            print(f"  总计: {total} 张")
    
    # 创建压缩包
    print("\n" + "=" * 60)
    print("创建压缩包...")
    print("=" * 60)
    
    output_file = "alzheimer-data-for-colab"
    
    # 使用 zip 压缩
    shutil.make_archive(
        output_file,
        'zip',
        root_dir='data',
        base_dir='splits'
    )
    
    zip_path = f"{output_file}.zip"
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    
    print(f"\n压缩包已创建: {zip_path}")
    print(f"文件大小: {size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("上传说明:")
    print("=" * 60)
    print("1. 将压缩包上传到 Google Drive")
    print("2. 在 Google Drive 中解压到: MyDrive/alzheimer-data/")
    print("3. 确保目录结构为: MyDrive/alzheimer-data/splits/train/...")
    print("\n或者直接在 Colab 中运行:")
    print("  !gdown --id YOUR_FILE_ID -O data.zip")
    print("  !unzip -q data.zip -d data/")
    print("=" * 60)

if __name__ == "__main__":
    prepare_data_for_upload()
