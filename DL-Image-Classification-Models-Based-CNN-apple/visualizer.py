import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
import torch
from PIL import Image, ImageTk
from torchvision import transforms
import pandas as pd
import threading

# 模型定义（需与训练代码一致）
from classic_models.mobilenet_v3 import mobilenet_v3_large
from classic_models.googlenet_v1 import GoogLeNet


class ClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("苹果分类检测系统")
        master.geometry("1200x800")

        weights_list = ['GoogleNet', 'MobileNetV3', 'AlexNet']

        # 初始化配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_path = f"./results/weights/{weights_list[0]}/{weights_list[0]}.pth"
        self.class_json_path = "./dataset/apple/class_indices.json"

        # 加载类别信息
        with open(self.class_json_path, "r") as f:
            self.class_dict = json.load(f)
        self.class_names = [self.class_dict[str(i)] for i in range(len(self.class_dict))]

        # 初始化模型
        # self.model = mobilenet_v3_large(num_classes=len(self.class_names)).to(self.device)
        self.model = GoogLeNet(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        # 创建GUI组件
        self.create_widgets()

        # 图像预处理
        self.data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
            # 存储预测结果
            file_results = []
            total_files = sum([len(files) for _, _, files in os.walk(self.test_folder)])
            processed = 0

            # 遍历文件
            for root, dirs, files in os.walk(self.test_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        file_path = os.path.join(root, file)

                        # 执行预测
                        img = Image.open(file_path).convert('RGB')
                        img_tensor = self.data_transform(img).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            output = self.model(img_tensor)
                            probs = torch.softmax(output, dim=1)
                            pred_class = torch.argmax(probs).item()
                            confidence = probs[0][pred_class].item()

                        # 记录结果
                        file_results.append({
                            "文件路径": file_path,
                            "预测类别": self.class_dict[str(pred_class)],
                            "置信度": f"{confidence:.2%}"
                        })

                        # 更新进度
                        processed += 1
                        self.update_progress((processed / total_files) * 100)

                        # 显示首张示例图片
                        if processed == 1:
                            self.show_sample_image(img, self.class_dict[str(pred_class)])

            # 显示最终统计
            self.show_final_statistics(file_results)

        except Exception as e:
            self.result_text.insert(tk.END, f"\n错误：{str(e)}")

    def show_sample_image(self, img, pred_class):
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.sample_label.configure(image=photo)
        self.sample_label.image = photo
        self.result_text.insert(tk.END, f"\n首张示例预测：{pred_class}")

    def show_final_statistics(self, results):
        # 统计类别分布
        df = pd.DataFrame(results)
        class_counts = df['预测类别'].value_counts()

        # 显示统计结果
        self.result_text.insert(tk.END, "\n\n预测结果统计：")
        for cls, count in class_counts.items():
            self.result_text.insert(tk.END, f"\n{cls}: {count}张 ({count / len(results):.1%})")

        # 导出按钮
        ttk.Button(self.master, text="导出完整结果",
                   command=lambda: self.export_results(results)).pack(side=tk.BOTTOM)

    def export_results(self, results):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if save_path:
            pd.DataFrame(results).to_csv(save_path, index=False)
            self.result_text.insert(tk.END, f"\n结果已保存至：{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ClassificationApp(root)
    root.mainloop()