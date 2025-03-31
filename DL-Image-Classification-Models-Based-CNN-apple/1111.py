import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
import torch
import numpy as np
import pandas as pds
from PIL import Image, ImageTk
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import threading
import matplotlib.pyplot as plt


# 模型定义（需与训练代码一致）
from classic_models.mobilenet_v3 import mobilenet_v3_large
from classic_models.googlenet_v1 import GoogLeNet
from classic_models.resnet import resnet101
from classic_models.mobilenet_v3pro import mobilenet_v3_large_CBAM
from classic_models.mobilenet_v3_ghost import mobilenet_v3_large_ghost


class ClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("苹果分类检测系统")
        master.geometry("1200x800")

        weights_list = ['GoogleNet', 'MobileNetV3', 'AlexNet',"ResNet","MobileNetV3_CBAM",'MobileNetV3_Ghost']

        # 初始化配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_path = f"./results/weights/{weights_list[5]}/{weights_list[5]}.pth"
        self.class_json_path = r"./dataset/apple/class_indices.json"

        # 加载类别信息
        with open(self.class_json_path, "r") as f:
            self.class_dict = json.load(f)
        self.class_names = [self.class_dict[str(i)] for i in range(len(self.class_dict))]

        # 初始化模型
        # self.model = mobilenet_v3_large(num_classes=len(self.class_names)).to(self.device)
        # self.model = mobilenet_v3_large_CBAM(num_classes=len(self.class_names)).to(self.device)
        self.model = mobilenet_v3_large_ghost(num_classes=len(self.class_names)).to(self.device)

        # self.model = GoogLeNet(num_classes=len(self.class_names)).to(self.device)
        # self.model = resnet101(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        # 创建GUI组件
        self.create_widgets()

        # 图像预处理
        self.data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def create_widgets(self):
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.master, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 文件夹选择
        ttk.Button(control_frame, text="选择测试文件夹", command=self.select_folder).pack(pady=5)

        # 结果显示区域
        self.result_text = tk.Text(control_frame, height=15, width=35)
        self.result_text.pack(pady=5)

        # 进度条
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        # 右侧展示区
        display_frame = ttk.Frame(self.master)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 混淆矩阵画布
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 示例图片展示
        self.sample_label = ttk.Label(display_frame)
        self.sample_label.pack()

    def select_folder(self):
        self.test_folder = filedialog.askdirectory()
        if self.test_folder:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"已选择文件夹：\n{self.test_folder}\n")
            threading.Thread(target=self.run_predictions).start()

    def update_progress(self, value):
        self.progress['value'] = value
        self.master.update_idletasks()

    def run_predictions(self):
        try:
            # 初始化结果存储
            all_preds = []
            all_true = []
            file_list = []

            # 获取文件总数
            total_files = sum([len(files) for _, _, files in os.walk(self.test_folder)])
            processed = 0

            # 遍历文件
            for root, dirs, files in os.walk(self.test_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        file_path = os.path.join(root, file)

                        # 获取真实标签
                        true_label = None
                        if root != self.test_folder:
                            dir_name = os.path.basename(root)
                            true_label = next((k for k, v in self.class_dict.items() if v == dir_name), None)

                        # 执行预测
                        img = Image.open(file_path).convert('RGB')
                        img_tensor = self.data_transform(img).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            output = self.model(img_tensor)
                            probs = torch.softmax(output, dim=1)
                            pred_class = torch.argmax(probs).item()

                        # 记录结果
                        if true_label is not None:
                            all_true.append(int(true_label))
                            all_preds.append(pred_class)

                        # 更新进度
                        processed += 1
                        self.update_progress((processed / total_files) * 100)

                        # 显示示例图片
                        if processed == 1:
                            self.show_sample_image(img, pred_class)

            # 显示最终结果
            self.show_final_results(all_true, all_preds)

        except Exception as e:
            self.result_text.insert(tk.END, f"\n错误：{str(e)}")

    def show_sample_image(self, img, pred_class):
        # 调整图片尺寸
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.sample_label.configure(image=photo)
        self.sample_label.image = photo
        self.result_text.insert(tk.END, f"\n示例预测：{self.class_dict[str(pred_class)]}")

    def show_final_results(self, all_true, all_preds):
        # 清空画布
        self.figure.clf()

        # 分类报告
        if len(all_true) > 0:
            report = classification_report(all_true, all_preds, target_names=self.class_names)
            self.result_text.insert(tk.END, "\n\n分类报告：\n" + report)

            # 混淆矩阵
            cm = confusion_matrix(all_true, all_preds)
            ax = self.figure.add_subplot(111)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            ax.set_title('Confusion Matrix')
            self.canvas.draw()
        else:
            self.result_text.insert(tk.END, "\n警告：未找到带标签的测试数据")

        # 导出按钮
        ttk.Button(self.master, text="导出结果", command=self.export_results).pack(side=tk.BOTTOM)

    def export_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            pd.DataFrame({
                '预测类别': [self.class_dict[str(p)] for p in self.all_preds],
                '真实类别': [self.class_dict[str(t)] for t in self.all_true]
            }).to_csv(save_path, index=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = ClassificationApp(root)
    root.mainloop()