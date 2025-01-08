import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


class ImageColirizationDataset(Dataset):
    def __init__(self,color_dir,gray_dir,transform=None):
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.color_images = sorted(os.listdir(color_dir))
        self.gray_images = sorted(os.listdir(gray_dir))
        self.transform = transform

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, item):
        color_image_path = os.path.join(self.color_dir,self.color_images[item])
        gray_image_path = os.path.join(self.gray_dir,self.gray_images[item])

        color_image = Image.open(color_image_path).convert('RGB')
        gray_image = Image.open(gray_image_path).convert('L')

        if self.transform:
            color_image = self.transform(color_image)
            gray_image = self.transform(gray_image)

        return gray_image, color_image


# 定义图像变换，compose将各种处理集合起来
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)), #调整尺寸，长宽比不变
        # transforms.CenterCrop(256),  # 裁剪图像
        transforms.ToTensor()    # 把一个取值范围[0,255]的PIL。Image转换成张量

    ]
)

color_dir = 'image_cg/color'
gray_dir = 'image_cg/gray'

datasets = ImageColirizationDataset(color_dir=color_dir, gray_dir=gray_dir, transform=transform)

dataloader = DataLoader(datasets, batch_size=32, shuffle=True, num_workers=4)


# 展示图片
def show_images(dataloader, num_images=3):
    data_iter = iter(dataloader)
    gray_imgs,color_imgs = next(data_iter)

    fig,axes = plt.subplots(num_images, 2, figsize=(10, num_images*5))
    for i in range(num_images):
        ax = axes[i, 0]
        gray_img = gray_imgs[i].numpy().transpose((1, 2, 0)).squeeze()
        ax.imshow(gray_img, cmap='gray')
        ax.set_title('Gray Image')
        ax.axis('off')

        ax = axes[i, 1]
        color_img = color_imgs[i].numpy().transpose((1, 2, 0))
        ax.imshow(color_img)
        ax.set_title('Color Image')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# 定义编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return  x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# print(f"using device {device}")
#
# model = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"using device {device}")

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, dataloader, criterion, optimizer,num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for gray_imgs, color_imgs in dataloader:
            gray_imgs, color_imgs = gray_imgs.to(device), color_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(gray_imgs)
            loss = criterion(outputs, color_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * gray_imgs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"epoch [{epoch+1}/{num_epochs}], LOSS:{epoch_loss:.4f}")


def visiulize_results(model, dataloader, num_images=5):
    model.eval()
    gray_imgs, color_imgs = next(iter(dataloader))

    gray_imgs = gray_imgs.to(device)
    with torch.no_grad():
        outputs = model(gray_imgs)
    gray_imgs, color_imgs, outputs = gray_imgs.cpu(), color_imgs.cpu(), outputs.cpu()

    fig,axes = plt.subplots(num_images, 3, figsize=(15, num_images*5))
    for i in range(num_images):
        # 灰度图
        ax = axes[i, 0]
        gray_img = gray_imgs[i].numpy().transpose((1, 2, 0)).squeeze()
        ax.imshow(gray_img, cmap='gray')
        ax.set_title('Gray Image')
        ax.axis('off')

        ax = axes[i, 1]
        color_img = color_imgs[i].numpy().transpose((1, 2, 0)).squeeze()
        ax.imshow(color_img)
        ax.set_title("Target Color Image")
        ax.axis('off')

        # 显示预测的彩图
        ax = axes[i, 2]
        output_img = outputs[i].numpy().transpose((1, 2, 0))
        ax.imshow(output_img)
        ax.set_title('Predicted color Image')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # show_images(lbw_dataloader)
    # model = Autoencoder()
    # print(model)
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # print(f"using device {device}")
    #
    # model = Autoencoder().to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    torch.manual_seed(666)
    train_model(model, dataloader,  criterion, optimizer, num_epochs=20)
    visiulize_results(model,dataloader,num_images=5)