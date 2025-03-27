# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 添加 tqdm 库

# 定义超参数
batch_size = 128
learning_rate = 0.002
num_epochs = 200
dprate = 0.5

# 改进数据增强的数据预处理方式
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加随机颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 改进模型结构
class Net(nn.Module):
    '''
    改进后的卷积神经网络，增加卷积层和批归一化
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, dprate)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, dprate)
        x = self.fc3(x)
        return x

# 实例化模型
model = Net()

use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 使用学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 每30个epoch学习率减半

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    train_cor = 0
    train_num = 0
    for i, (images, labels) in train_pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_num += labels.size(0)
        train_cor += (outputs.argmax(1) == labels).sum().item()
        accuracy = train_cor / train_num

        # 更新进度条信息
        train_pbar.set_postfix({"Loss": loss.item(), "Accuracy": f"{accuracy * 100:.2f}%"})

    scheduler.step()  # 更新学习率

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
        for images, labels in test_pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条信息
            test_pbar.set_postfix({"Accuracy": f"{100 * correct / total:.2f}%"})

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))