"""
通用数据集划分工具(训练集和测试集)
"""
import random
import shutil
from pathlib import Path

# 支持的图片格式
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']


def split_dataset(source_dir, output_dir, train_ratio=0.8, random_seed=42):
    random.seed(random_seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 自动发现所有类别文件夹
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    class_names = [d.name for d in class_dirs]
    
    if not class_names:
        print(f"❌ 在源目录中没有找到任何子文件夹")
        return
    
    print(f"发现 {len(class_names)} 个类别: {', '.join(class_names)}")
    
    # 创建输出目录结构
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    
    for split_dir in [train_dir, test_dir]:
        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个类别
    for class_name in class_names:
        source_class_dir = source_path / class_name
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(source_class_dir.glob(f"*{ext}")))
        
        if not image_files:
            print(f"⚠️ {class_name} 目录中没有找到图片文件")
            continue
        
        # 打乱文件列表
        random.shuffle(image_files)
        
        # 计算划分点
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        
        print(f"处理 '{class_name}' 类别: 总数{total_images}, 训练{train_count}, 测试{total_images-train_count}")
        
        # 复制文件到训练集
        train_class_dir = train_dir / class_name
        for img_file in image_files[:train_count]:
            dest_file = train_class_dir / img_file.name
            shutil.copy2(img_file, dest_file)
        
        # 复制文件到测试集
        test_class_dir = test_dir / class_name
        for img_file in image_files[train_count:]:
            dest_file = test_class_dir / img_file.name
            shutil.copy2(img_file, dest_file)
    
    print(f"\n✅ 完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    # 配置参数
    source_directory = "./data/cat_dog"  # 原始数据集路径
    output_directory = "./dataset_split"  # 输出路径
    train_ratio = 0.8  # 训练集比例
    random_seed = 42 #随机种子
    
    # 执行数据集划分
    split_dataset(
        source_dir=source_directory,
        output_dir=output_directory,
        train_ratio=train_ratio,
        random_seed = random_seed
    )
