import os
from datetime import datetime


def rename_images_to_numbers(folder_path, sort_by='name'):
    # 定义支持的图片扩展名
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    # 获取所有图片文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_exts)]

    # 排序方式
    if sort_by == 'name':
        files.sort()  # 按文件名排序
    elif sort_by == 'modified':
        files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))  # 按修改时间排序
    elif sort_by == 'created':
        files.sort(key=lambda f: os.path.getctime(os.path.join(folder_path, f)))  # 按创建时间排序

    # 开始重命名
    for idx, filename in enumerate(files, start=1):
        # 原文件的完整路径
        old_path = os.path.join(folder_path, filename)

        # 提取扩展名（保留原扩展名大小写）
        _, ext = os.path.splitext(filename)

        # 生成新文件名（如 "1.jpg", "2.png"）
        new_name = f"{idx}{ext}"
        new_path = os.path.join(folder_path, new_name)

        # 避免重名冲突：如果新文件名已存在，则跳过
        if os.path.exists(new_path):
            print(f"跳过 {filename} → {new_name}（文件已存在）")
            continue

        # 执行重命名
        os.rename(old_path, new_path)
        print(f"重命名 {filename} → {new_name}")


# 使用方法
folder = r"D:\post-file\DL-Image-Classification-Models-Based-CNN\dataset_unit\apple\val\apple_1"
rename_images_to_numbers(folder, sort_by='name')  # 按文件名排序
