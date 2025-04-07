# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
import sys

from utils import load_imdb_dataset, Accuracy
from tqdm import tqdm

use_mlu = False
# try:
#     import torch_mlu
#     import torch_mlu.core.mlu_model as ct
#     global ct
#     use_mlu = torch.mlu.is_available()
# except:
#     use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1


class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)


# 你需要实现的手写LSTM内容，包括LSTM类所属的__init__函数和forward函数
class LSTM(nn.Module):
    '''
    手写lstm，可以用全连接层nn.Linear，不能直接用nn.LSTM
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        # LSTM层
        # 单层LSTM
        self.Wii = nn.Linear(input_size, hidden_size, bias=True) # 输入门
        self.Wif = nn.Linear(input_size, hidden_size, bias=True) # 遗忘门
        self.Wig = nn.Linear(input_size, hidden_size, bias=True) # 候选记忆单元
        self.Wio = nn.Linear(input_size, hidden_size, bias=True) # 输出门
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=True) # 隐藏状态输入门
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=True) # 隐藏状态遗忘门
        self.Whg = nn.Linear(hidden_size, hidden_size, bias=True) # 隐藏状态候选记忆单元
        self.Who = nn.Linear(hidden_size, hidden_size, bias=True) # 隐藏状态输出门

    def forward(self, x, h_prev, c_prev):
        '''
        x: 输入, shape: (batch_size, seq_len, input_size)
        h_prev: 上一时刻的隐藏状态, shape: (batch_size, hidden_size)
        c_prev: 上一时刻的细胞状态, shape: (batch_size, hidden_size)
        '''
        outputs = []  # 用于存储每个时间步的隐藏状态
        for t in range(x.size(1)):  # 遍历序列长度
            x_t = x[:, t, :]  # 当前时间步的输入 (batch_size, input_size)
            i_t = torch.sigmoid(self.Wii(x_t) + self.Whi(h_prev))
            f_t = torch.sigmoid(self.Wif(x_t) + self.Whf(h_prev))
            g_t = torch.tanh(self.Wig(x_t) + self.Whg(h_prev))
            o_t = torch.sigmoid(self.Wio(x_t) + self.Who(h_prev))
            c_prev = f_t * c_prev + i_t * g_t  # 更新细胞状态
            h_prev = o_t * torch.tanh(c_prev)  # 更新隐藏状态
            outputs.append(h_prev.unsqueeze(1))  # 添加当前时间步的隐藏状态

        outputs = torch.cat(outputs, dim=1)  # 拼接所有时间步的隐藏状态 (batch_size, seq_len, hidden_size)
        return outputs, h_prev, c_prev  # 返回所有时间步的隐藏状态、最后一个隐藏状态和细胞状态


# 你需要实现网络推理和训练内容，仅需要完善forward函数
class Net(nn.Module):
    '''
    一层LSTM的文本分类模型
    '''

    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM层
        self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size)
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        '''
        x: 输入, shape: (batch_size, seq_len)
        '''

        # 词嵌入
        x = self.embedding(x)  # (batch_size, seq_len, embedding_size)
        
        # 初始化隐藏状态和细胞状态
        batch_size_in_forward = x.shape[0]
        h_prev = torch.zeros(batch_size_in_forward, self.lstm.hidden_size).to(device)  # (batch_size, hidden_size)
        c_prev = torch.zeros(batch_size_in_forward, self.lstm.hidden_size).to(device)  # (batch_size, hidden_size)
        
        # LSTM层逐时间步计算
        outputs, h_prev, c_prev = self.lstm(x, h_prev, c_prev)
        
        # 全连接层
        x = torch.relu(self.fc1(h_prev))  # 使用最后一个时间步的隐藏状态
        x = torch.softmax(self.fc2(x), dim=1)
        return x


n_epoch = 5
batch_size = 64
print_freq = 2

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = Net()
metric = Accuracy()
print(net)


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output, target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, train_loss / n_iter, train_acc / n_iter))


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    test_loss = 0
    test_acc = 0
    n_iter = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_func(output, target).item()
            metric.update(output, target)
            test_acc += metric.result()
            metric.reset()
            n_iter += 1
    test_loss /= n_iter
    test_acc /= n_iter
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
