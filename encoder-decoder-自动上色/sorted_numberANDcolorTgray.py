import os
import shutil
from PIL import Image


def rename_images(file_path, gray_path):
    files = os.listdir(file_path)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 将文件名转换为整数并排序
    sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    # 重命名文件
    for idx, filename in enumerate(sorted_files):
        new_filename = f"{idx}.{filename.split('.')[-1]}"
        old_filepath = os.path.join(file_path, filename)
        new_filepath = os.path.join(file_path, new_filename)
        shutil.move(old_filepath, new_filepath)
        print(f"sorted {old_filepath} to {new_filepath}")

        color_image = Image.open(new_filepath)
        print(color_image)
        gray_image = color_image.convert('L')
        gray_image.save(os.path.join(gray_path,new_filename))
        print(f"successfully convert {new_filename} to grayimg")
        # break


file_path = r"D:\python-workplace_AC\machine_learning\DT\深度学习\encoder-decoer-自动着色器\image_cg\color"
gray_path = r"D:\python-workplace_AC\machine_learning\DT\深度学习\encoder-decoer-自动着色器\image_cg\gray"
rename_images(file_path, gray_path)

