# -*- coding=utf-8 -*-
# 文件名: predict.py
# 作者: Theia-4869 (Qizhe Zhang)
# 功能: 本文件用于火车票票面检测与序列号识别的最终预测


import os
import math
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def predict(input_dir, annotation_file, output_dir, prediction_file):
    """
    [predict函数用于序列号识别预测]

    Args:
        input_dir ([str]): [数字与字母图片所在文件夹]
        annotation_file ([str]): [注释文件名]
        output_dir ([str]): [票面检测输出文件夹]
        prediction_file ([str]): [预测输出文件名]
    """
    # 为输出票面检测结果创建目录
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # 读取预训练模型
    model_dir = "models"
    ckps = os.listdir(model_dir)
    ckp = ckps[-1]

    # 创建模型, 载入预训练模型参数
    model = SimpleCNN(number_class=10, letter_class=26)
    print("Loading model:", ckp)
    model.load_state_dict(torch.load(os.path.join(model_dir, ckp)), strict=False)
    model.eval()

    # 打开注释文件与输出文件
    infile = open(annotation_file,'r')
    outfile = open(prediction_file,'w')
    lines = infile.readlines()

    '''
    [预测部分]
    '''
    print("Predicting...")
    for line in lines:
        filename = line.split()[0]
        if filename[-4:] != ".bmp" and filename[-4:] != ".jpg" and filename[-4:] != ".png":
            continue    # 对于目录中的非图片文件, 不予理会

        '''
        [票面检测部分]
        '''
        # 读入原始图片, 灰度化, 二值化
        original_img = cv2.imread(os.path.join(input_dir, filename))
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
        
        # 中值滤波去噪
        median_img = cv2.medianBlur(binary_img, 7)
        
        # 形态学处理 (开-闭-开) 去除票面上的文字与线条
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 70))
        mor_img = cv2.morphologyEx(median_img, cv2.MORPH_OPEN, kernel1) 
        mor_img = cv2.morphologyEx(mor_img, cv2.MORPH_CLOSE, kernel2)
        mor_img = cv2.morphologyEx(mor_img, cv2.MORPH_OPEN, kernel2)  
        
        # 绘制票面最小矩形包围盒
        coords = np.column_stack(np.where(mor_img > 0))
        rect = cv2.minAreaRect(coords)
        box_ = np.int0(cv2.boxPoints(rect))
        box = np.zeros(box_.shape, dtype=np.int64)
        box[:, 0] = box_[:, 1]
        box[:, 1] = box_[:, 0]
        box_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 2)
        
        # 票面转正
        h, w = mor_img.shape
        cx, cy = w/2, h/2
        if rect[1][0] < rect[1][1]:
            angle = -rect[2]
        else:
            angle = -rect[2] - 90
        M = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=1.0)
        new_h = int(w * math.fabs(math.sin(math.radians(angle))) + h * math.fabs(math.cos(math.radians(angle))))
        new_w = int(h * math.fabs(math.sin(math.radians(angle))) + w * math.fabs(math.cos(math.radians(angle))))
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        upright_img = cv2.warpAffine(original_img, M, (new_w, new_h), borderValue=(0, 0, 0))
        upright_mor_img = cv2.warpAffine(mor_img, M, (new_w, new_h), borderValue=(0, 0, 0))
        upright_box_img = cv2.warpAffine(box_img, M, (new_w, new_h), borderValue=(0, 0, 0))
        
        # 判断票面是否倒置, 若是则翻转180°
        gray_upright_img = cv2.cvtColor(upright_img, cv2.COLOR_BGR2GRAY)
        plus_img = cv2.threshold(gray_upright_img, 127, 255, cv2.THRESH_BINARY)[1] - cv2.threshold(gray_upright_img, 200, 255, cv2.THRESH_BINARY)[1]
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        plus_img = cv2.erode(plus_img, erode_kernel)
        if np.sum(plus_img[:h//2, :]) > np.sum(plus_img[h//2:, :]):
            upright_img = cv2.flip(upright_img, -1)
            upright_mor_img = cv2.flip(upright_mor_img, -1)
            upright_box_img = cv2.flip(upright_box_img, -1)
        
        # 将票面单独分割出来并统一缩放为 1080×640 大小
        coords = np.column_stack(np.where(upright_mor_img > 0))
        rect = cv2.minAreaRect(coords)
        box = np.int0(cv2.boxPoints(rect))
        h_min = np.min(box[:, 0])
        h_max = np.max(box[:, 0])
        w_min = np.min(box[:, 1])
        w_max = np.max(box[:, 1])
        select_img = upright_img[h_min:h_max, w_min:w_max]
        face_img = cv2.resize(select_img, (1080, 640), interpolation=cv2.INTER_CUBIC) 

        '''
        [21位码定位与分割]
        '''
        # 灰度化, 高斯模糊, 二值化, 初步定位21位码大致区域
        gray_img_21 = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gaussian_img_21 = cv2.GaussianBlur(gray_img_21, (5, 5), 0)
        binary_img_21 = cv2.threshold(gaussian_img_21, 30, 255, cv2.THRESH_BINARY)[1]
        mask_21 = np.ones(binary_img_21.shape, dtype=np.uint8)
        mask_21[500:620, 20:800] = 0
        img_21 = cv2.add(binary_img_21, mask_21 * 255)
        
        # 以区域左下角为基准点粗定位21位码
        coords_21 = np.column_stack(np.where(img_21 < 255))
        rect_21 = cv2.minAreaRect(coords_21)
        box_21_ = np.int0(cv2.boxPoints(rect_21))
        box_21_ = box_21_[np.lexsort(np.rot90(box_21_))]
        h_max_21 = np.max(box_21_[:, 0])
        w_min_21 = np.min(box_21_[:, 1])
        img_21[:h_max_21-40, :] = 255
        img_21[:, w_min_21+405:] = 255
        
        # 以粗定位为基准精细定位21位码
        coords_21 = np.column_stack(np.where(img_21 < 255))
        rect_21 = cv2.minAreaRect(coords_21)
        box_21_ = np.int0(cv2.boxPoints(rect_21))
        box_21_ = box_21_[np.lexsort(np.rot90(box_21_))]
        h_min_21 = np.min(box_21_[:, 0]) - 3
        h_max_21 = np.max(box_21_[:, 0]) + 3
        w_min_21 = np.min(box_21_[:, 1]) - 4
        w_max_21 = np.max(box_21_[:, 1]) + 4
        box_21 = np.array([[w_min_21, h_min_21], [w_min_21, h_max_21], [w_max_21, h_max_21], [w_max_21, h_min_21]], dtype=np.int64)
        face_img_21 = cv2.drawContours(face_img.copy(), [box_21], -1, (0, 0, 255), 3)
        face_img_21_copy = face_img_21.copy()
        
        # 用掩模提取21位码区域以待分割
        mask_21 = np.ones(img_21.shape, dtype=np.uint8)
        mask_21[h_min_21:h_max_21, w_min_21:w_max_21] = 0
        mask_box_21 = np.add(img_21,  mask_21 * 255)
        
        # 根据连通域以自适应间距精细分割21位码
        contours = cv2.findContours(mask_box_21, 2, 2)[0]
        x_min = 2000
        x_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if x != 0 and y != 0 and w*h >= 40:
                x_list.append(x)
                if x < x_min:
                    x_min = x
        x_list.sort()
        x_list.pop(0)
        y_list = [w_min_21+4]
        line_num_21 = 0
        for i in range(len(x_list)):
            if line_num_21 == 14:
                dist_min = 20
            elif line_num_21 == 13:
                x_list[i] -= 2
            else:
                dist_min = 10
            if x_list[i] > x_min + dist_min:
                cv2.line(face_img_21_copy, (x_list[i]-1, h_min_21), (x_list[i]-1, h_max_21), (0, 0, 255), 2)
                x_min = x_list[i]
                y_list.append(x_list[i])
                line_num_21 += 1
        y_list.append(w_max_21)
        if line_num_21 == 20:   # 若21位都被单独分离, 则分割完成
            face_img_21 = face_img_21_copy
        else:                   # 否则采用等距分割以保证程序鲁棒性
            interval = (w_max_21 - w_min_21 - 6) / 21.5
            dist = 0
            y_list = [w_min_21+4]
            for i in range(20):
                if i == 14:
                    dist += interval * 1.5
                else:
                    dist += interval
                cv2.line(face_img_21, (w_min_21+3+np.int0(dist), h_min_21), (w_min_21+3+np.int0(dist), h_max_21), (0, 0, 255), 2)
                y_list.append(w_min_21+4+np.int0(dist))
            y_list.append(w_max_21-2)
        
        '''
        [21位码预测]
        '''
        decode_21 = " "
        for i in range(21):
            # 单个数字或字母提取, 统一缩放为 32×32 大小, 灰度化, 二值化, 转换为神经网络输入张量格式
            code_21 = face_img[h_min_21:h_max_21, y_list[i]-2:y_list[i+1]-1]
            code_21 = cv2.resize(code_21, (32, 32), interpolation=cv2.INTER_CUBIC)
            code_21 = cv2.cvtColor(code_21, cv2.COLOR_BGR2GRAY)
            code_21 = cv2.threshold(code_21, 30, 255, cv2.THRESH_BINARY)[1]
            code_21 = torch.from_numpy(code_21)
            code_21 = code_21.unsqueeze(0).unsqueeze(0).float()
            
            # 分数字与字母不同模式分别预测
            if i == 14:
                logits = model(code_21, "letter")
                predict = logits.max(dim=1)[1]
                decode_21 += chr(predict+65)
            else:
                logits = model(code_21, "number")
                predict = logits.max(dim=1)[1]
                decode_21 += str(int(predict[0]))

        '''
        [7位码定位与分割]
        '''
        # 灰度化, 双阈值二值化
        gray_img_7 = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        binary_img_7 = cv2.threshold(gray_img_7, 50, 255, cv2.THRESH_BINARY)[1] - cv2.threshold(gray_img_7, 135, 255, cv2.THRESH_BINARY)[1]
        
        # 形态学操作 (腐蚀-闭) 去噪
        erode_kernel_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mor_img_7 = cv2.erode(binary_img_7, erode_kernel_7)
        close_kernel_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mor_img_7 = cv2.morphologyEx(mor_img_7, cv2.MORPH_CLOSE, close_kernel_7)
        
        # 在票面左上角粗定位7位码
        mask_7 = np.zeros(mor_img_7.shape, dtype=np.uint8)
        mask_7[10:200, 20:400] = 1
        mask_img_7 = mor_img_7 * mask_7
        
        # 根据连通域大小进一步去噪
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask_img_7, connectivity=8)
        img_7 = mask_img_7.copy()
        for i in range(retval):
            if stats[i][4] < 150:
                img_7[labels==i] = 0
        
        # 以粗定位为基准精细定位7位码
        coords_7 = np.column_stack(np.where(img_7 > 0))
        rect_7 = cv2.minAreaRect(coords_7)
        box_7_ = np.int0(cv2.boxPoints(rect_7))
        box_7_ = box_7_[np.lexsort(np.rot90(box_7_))]
        h_min_7 = np.min(box_7_[:, 0]) - 3
        h_max_7 = np.max(box_7_[:, 0]) + 3
        w_min_7 = np.min(box_7_[:, 1]) - 4
        w_max_7 = np.max(box_7_[:, 1]) + 4
        box_7 = np.array([[w_min_7, h_min_7], [w_min_7, h_max_7], [w_max_7, h_max_7], [w_max_7, h_min_7]], dtype=np.int64)
        face_img_7 = cv2.drawContours(face_img_21, [box_7], -1, (0, 0, 255), 3)
        
        # 用掩模提取7位码区域以待分割
        binary_img_7 = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        binary_img_7 = cv2.medianBlur(binary_img_7, 3)
        binary_img_7 = cv2.threshold(binary_img_7, 50, 255, cv2.THRESH_BINARY)[1] - cv2.threshold(binary_img_7, 150, 255, cv2.THRESH_BINARY)[1]
        binary_img_7 = 255 - binary_img_7
        mask_7 = np.ones(binary_img_7.shape, dtype=np.uint8)
        mask_7[h_min_7:h_max_7, w_min_7:w_max_7] = 0
        mask_box_7 = binary_img_7 + mask_7 * 255
        face_img_21_copy = face_img_21.copy()
        
        # 根据连通域以自适应间距精细分割7位码
        contours = cv2.findContours(mask_box_7, 2, 2)[0]
        x_min = 2000
        x_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if x != 0 and y != 0 and w*h >= 200:
                x_list.append(x)
                if x < x_min:
                    x_min = x
        x_list.sort()
        x_list.pop(0)
        y_list = [w_min_7+3]
        line_num_7 = 0
        for i in range(len(x_list)):
            if x_list[i] > x_min + 20:
                cv2.line(face_img_21_copy, (x_list[i]-1, h_min_7), (x_list[i]-1, h_max_7), (0, 0, 255), 2)
                x_min = x_list[i]
                y_list.append(x_list[i])
                line_num_7 += 1
        y_list.append(w_max_7-3)
        if line_num_7 == 6:     # 若7位都被单独分离, 则分割完成
            face_img_21 = face_img_21_copy
        else:                   # 否则采用等距分割以保证程序鲁棒性
            interval = (w_max_7 - w_min_7 - 6) / 7.5
            dist = 0
            y_list = [w_min_7+3]
            for i in range(6):
                if i == 0:
                    dist += interval * 1.5
                else:
                    dist += interval
                cv2.line(face_img_21, (w_min_7+3+np.int0(dist), h_min_7), (w_min_7+3+np.int0(dist), h_max_7), (0, 0, 255), 2)
                y_list.append(w_min_7+3+np.int0(dist))
            y_list.append(w_max_7)

        '''
        [7位码预测]
        '''
        decode_7 = " "
        for i in range(7):
            # 单个数字或字母提取, 统一缩放为 32×32 大小, 灰度化, 二值化, 转换为神经网络输入张量格式
            code_7 = face_img[h_min_7:h_max_7, y_list[i]-2:y_list[i+1]-1]
            code_7 = cv2.resize(code_7, (32, 32), interpolation=cv2.INTER_CUBIC)
            code_7 = cv2.threshold(code_7, 150, 255, cv2.THRESH_BINARY)[1]
            code_7 = cv2.cvtColor(code_7, cv2.COLOR_BGR2GRAY)
            code_7 = torch.from_numpy(code_7)
            code_7 = code_7.unsqueeze(0).unsqueeze(0).float()
            
            # 分数字与字母不同模式分别预测
            if i == 0:
                logits = model(code_7, "letter")
                predict = logits.max(dim=1)[1]
                decode_7 += chr(predict+65)
            else:
                logits = model(code_7, "number")
                predict = logits.max(dim=1)[1]
                decode_7 += str(int(predict[0]))

        '''
        [序列码预测后处理]
        '''
        # 若7位码与21位码后半部分预测结果不同, 则数字采用21位码预测结果, 字母采用7位码预测结果
        # 这是由于7位码部分可能与上方文字重合影响识别, 但由于尺寸较大定位较精准, 且字母处于第1位受影响较小
        # 而21位码虽然相对清晰, 但字母与数字尺寸相近, 对字母的分割可能存在不精准的情况
        if decode_7 != " "+decode_21[-7:]:
            new_decode_7 = decode_7[0:2]+decode_21[-6:]
            decode_7 = new_decode_7
        outfile.write(filename+decode_21+decode_7+"\n")
        
        # 写入预测文件
        cv2.imwrite(os.path.join(output_dir, filename), face_img_21)

    # 关闭注释文件与输出文件
    infile.close()
    outfile.close()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction for DIP Final Project')
    parser.add_argument('--input-dir', default='test_data', type=str)
    parser.add_argument('--annotation-file', default='annotation.txt', type=str)
    parser.add_argument('--output-dir', default='segments', type=str)
    parser.add_argument('--prediction-file', default='prediction.txt', type=str)
    args = parser.parse_args()
    
    predict(args.input_dir, args.annotation_file, args.output_dir, args.prediction_file)
