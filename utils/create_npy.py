import numpy as np
from PIL import Image
import csv
import os


def create_npy_dataset(cars_root_dir, output_filename):
    # 定义路径
    data_dir = os.path.join(cars_root_dir, 'data')
    csv_path = os.path.join(cars_root_dir, 'splits', 'default', 'train.csv')

    # 读取CSV文件
    filenames = []
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)  # 使用DictReader自动处理header
        for row in reader:
            filenames.append(row['filename'])
            labels.append(str(row['label']))  # 转换为字符串

    # 初始化数据存储
    num_images = len(filenames)
    image_data = np.zeros((num_images, 84, 84, 3), dtype=np.uint8)

    # 处理并存储图像
    for i, filename in enumerate(filenames):
        img_path = os.path.join(data_dir, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((84, 84))
            image_data[i] = np.array(img)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # 如果出现错误，用全黑图像填充
            image_data[i] = np.zeros((84, 84, 3), dtype=np.uint8)

    # 创建字典
    dataset = {
        'data': image_data,
        'labels': np.array(labels)
    }

    # 保存为npy文件
    np.save(output_filename, dataset)
    print(f"Dataset saved to {output_filename} with shape {image_data.shape}")


# 使用示例
create_npy_dataset('../datasets/cub', '../data/birds_train_data.npy')