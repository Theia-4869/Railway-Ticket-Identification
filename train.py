# -*- coding=utf-8 -*-
# 文件名: train.py
# 作者: Theia-4869 (Qizhe Zhang)
# 功能: 本文件车票序列号中的数字与字母识别的模型训练


import os
import argparse
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


class ticketDataset(Dataset):
    """
    [ticketDataset:数字与字母统一的数据集]
    """
    def __init__(
        self, 
        dir: str, 
        mode: str = 'train',
        category: str = 'number',
    ):
        super(ticketDataset, self).__init__()
        self.dir = dir
        self.mode = mode
        self.category = category
        self.ids = [os.path.join(dir, mode, filename) for filename in os.listdir(os.path.join(dir, mode))]
        if not self.ids:
            raise RuntimeError(f'No input file found in {os.path.join(dir, mode)}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.ids[idx], cv2.IMREAD_GRAYSCALE)
        label = self.ids[idx].split("\\")[2].split("_")[0]
        if self.category == "number":
            label = int(label)
        elif self.category == "letter":
            label = ord(label)-65
        return image, label

class SimpleCNN(nn.Module):
    """
    [SimpleCNN: 简单的CNN网络, 数字与字母共用最后一层之前的网络参数]

    Architecture:
        conv1: in_channels=1, out_channels=10, kernel_size=5
        pool: kernel_size=2, stride=2
        conv2: in_channels=10, out_channels=20, kernel_size=3
        flatten: flatten
        fc: in_features=20*12*12, out_features=512
        fc_number: in_features=512, out_features=number_class (10)
        fc_letter: in_features=512, out_features=letter_class (26)
    """
    def __init__(self, number_class=10, letter_class=26):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)    # (1, 32, 32) -> (10, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                  # (10, 28, 28) -> (10, 14, 14)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)   # (10, 14, 14) -> (20, 12, 12)
        self.flatten = nn.Flatten()                     # (20, 12, 12) -> (20*12*12)
        self.fc = nn.Linear(20*12*12, 512)              # (20*12*12) -> (512)
        self.fc_number = nn.Linear(512, number_class)   # (512) -> (number_class=10)
        self.fc_letter = nn.Linear(512, letter_class)   # (512) -> (letter_class=26)

    def forward(self, x, type="number"):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = F.relu(out)
        if type == "number":
            logits = self.fc_number(out)
        elif type == "letter":
            logits = self.fc_letter(out)
        return logits

def train(model_dir, learning_rate, batch_size, epoch):
    """
    [train函数用于训练模型]

    Args:
        model_dir ([str]): [模型存储文件夹]]
        learning_rate ([float]): [学习率]
        batch_size ([int]): [批大小]
        epoch ([int]): [训练轮数]]
    """
    # 为存储模型创建目录
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 创建数字与字母训练集、验证集
    number_trainset = ticketDataset("number_data", "train", "number")
    number_valset = ticketDataset("number_data", "val", "number")
    letter_trainset = ticketDataset("letter_data", "train", "letter")
    letter_valset = ticketDataset("letter_data", "val", "letter")
    
    number_trainloader = DataLoader(number_trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    number_valloader = DataLoader(number_valset, batch_size=batch_size, shuffle=True, drop_last=True)
    letter_trainloader = DataLoader(letter_trainset, batch_size=batch_size, shuffle=True)
    letter_valloader = DataLoader(letter_valset, batch_size=batch_size, shuffle=True)

    # 设置是否使用GPU, 创建模型, Adam优化器
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(number_class=10, letter_class=26).to(device=DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    max_accuracy = 0
    max_accuracy_number = 0
    max_accuracy_letter = 0

    '''
    [训练部分]
    '''
    print("Training...")
    for i in range(epoch):
        print("Epoch %d:, best_accuracy: %.5f" % (i+1, max_accuracy))
        
        # 数字识别训练
        model.train()
        number_loss = 0
        tbar = tqdm(number_trainloader)
        for j, (image, label) in enumerate(tbar):
            image = image.unsqueeze(1).float().to(device=DEVICE)
            label = label.to(device=DEVICE)
            logits = model(image, "number")
            loss = F.cross_entropy(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            number_loss += loss.item()
            tbar.set_description("Number train loss: %.5f" % (number_loss / (j + 1)))
        
        # 数字识别验证
        model.eval()
        number_accuracy = 0
        tbar = tqdm(number_valloader)
        for j, (image, label) in enumerate(tbar):
            image = image.unsqueeze(1).float().to(device=DEVICE)
            label = label.to(device=DEVICE)
            logits = model(image, "number")
            predict = logits.max(dim=1)[1]
            accuracy = (predict == label).sum() / label.shape[0]
            number_accuracy += accuracy.item()
            tbar.set_description("Number val accuracy: %.5f" % (number_accuracy / (j + 1)))
            max_accuracy_number = number_accuracy / (j + 1)
        
        # 字母识别训练
        model.train()
        letter_loss = 0
        tbar = tqdm(letter_trainloader)
        for j, (image, label) in enumerate(tbar):
            image = image.unsqueeze(1).float().to(device=DEVICE)
            label = label.to(device=DEVICE)
            logits = model(image, "letter")
            loss = F.cross_entropy(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            number_loss += loss.item()
            tbar.set_description("Letter train loss: %.5f" % (number_loss / (j + 1)))
        
        # 字母识别验证
        model.eval()
        letter_accuracy = 0
        tbar = tqdm(letter_valloader)
        for j, (image, label) in enumerate(tbar):
            image = image.unsqueeze(1).float().to(device=DEVICE)
            label = label.to(device=DEVICE)
            logits = model(image, "letter")
            predict = logits.max(dim=1)[1]
            accuracy = (predict == label).sum() / label.shape[0]
            letter_accuracy += accuracy.item()
            tbar.set_description("Letter val accuracy: %.5f" % (letter_accuracy / (j + 1)))
            max_accuracy_letter = letter_accuracy / (j + 1)
        
        # 比较准确率并存储模型
        accuracy = max_accuracy_number * max_accuracy_letter
        if accuracy > max_accuracy - 1e-5:
                delete_path = os.path.join(model_dir, "best_%.5f.pth" % max_accuracy)
                if os.path.exists(delete_path):
                    os.remove(delete_path)
                max_accuracy =accuracy
                save_path = os.path.join(model_dir, "best_%.5f.pth" % max_accuracy)
                torch.save(model.state_dict(), save_path)
    print("Done!")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for DIP Final Project')
    parser.add_argument('--model-dir', default='models', type=str)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    args = parser.parse_args()
    print(args)
    
    train(args.model_dir, args.learning_rate, args.batch_size, args.epoch)
