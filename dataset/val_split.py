import pandas as pd
import os
import shutil

def organize_images(csv_path, image_dir, save_dir):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 创建保存目录
    save_images_dir = os.path.join(save_dir, 'val_images')
    os.makedirs(save_images_dir, exist_ok=True)

    # 定义类别
    categories = ["healthy", "scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew"]

    # 创建每个类别的子目录
    for category in categories:
        category_dir = os.path.join(save_images_dir, category)
        os.makedirs(category_dir, exist_ok=True)

    label_columns = df.columns[1:]

    # 将图片复制到相应的类别目录
    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, row['images'])

        # 检查除了第一列之外的其他列是否为1，如果是，则复制图片
        for label_column in label_columns:
            if row[label_column] == 1:
                destination_dir = os.path.join(save_images_dir, label_column)
                shutil.copy(image_path, destination_dir)

    print("Images organized successfully.")

# 输入文件路径和保存目录
csv_path = "/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/dataset/val.csv"
image_dir = "/home/hbenke/Project/Lvwc/Project/Data/leaf_disease/val/images"
save_dir = "/home/hbenke/Project/Lvwc/Project/Data"

organize_images(csv_path, image_dir, save_dir)
