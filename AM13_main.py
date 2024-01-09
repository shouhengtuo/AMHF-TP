#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/26 21:04
# @Author : fhh
# @FileName: model.py
# @Software: PyCharm

import datetime
import os
import csv
import re
import numpy as np
import pandas as pd
from model import *
from torch.utils.data import DataLoader
from loss_functions import *
from train import CosineScheduler
# from swin_transformer import SwinTransformerModel
from ChouFasman import ChouFasman
from AM13_train import *
from AM13_model import *
from seq_HyGAN import get_hyG
torch.autograd.set_detect_anomaly(True)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']


def PadEncode(data, label, max_len):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e), np.array(seq_length)


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label


def staticTrainAndTest(y_train, y_test):
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    return data_size_tr


def main(num, data):
    # 获得图结构数据
    # 将测试数据与训练数据分开
    # sequence_X, train_data = get_hyG("./dataset/hyG_train.txt")
    # _, test_data = get_hyG("./dataset/hyG_test.txt")
    # 将全部数据都用来构造图数据，不分开测试数据与训练数据
    sequence_X, train_data = get_hyG("./dataset/target.txt")
    # 初始化模型
    model = Seq_HyGAN(sequence_X.shape[1], 64, 128, 128, 21, 0.3)
    rate_learning = data['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
    criterion = nn.BCEWithLogitsLoss()
    Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)
    # 训练模型（保存验证结果最好的模型），返回保存的最好的预测结果
    a = time.time()
    best_test_pred, test_label = Train.train_step(train_data, epochs=data['epochs'], plot_picture=False)
    b = time.time()
    # 计算训练时间
    runtime = b - a
    # 加载保存的模型进行测试
    best_test_pred, test_label = Train.load_model_test(model, train_data)
    test_score = hyG_evaluate(model, best_test_pred, test_label, device=DEVICE)

    # "-------------------------------------------保存模型参数-----------------------------------------------"
    # PATH = os.getcwd()
    # each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
    # torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
    # "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------输出模型结果-----------------------------------------------"
    print(f"runtime:{runtime:.3f}s")
    print("测试集：")
    print(f'aiming: {test_score["aiming"]:.3f}')
    print(f'coverage: {test_score["coverage"]:.3f}')
    print(f'accuracy: {test_score["accuracy"]:.3f}')
    print(f'absolute_true: {test_score["absolute_true"]:.3f}')
    print(f'absolute_false: {test_score["absolute_false"]:.3f}')
    "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------保存模型结果-----------------------------------------------"
    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
    # 这个值需要修改三个地方，除了这里还有HyG_train文件中保存模型的方法，加载模型的方法中
    model_name = "AM13_1"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [[model_name, '%.3f' % test_score["aiming"],
                '%.3f' % test_score["coverage"],
                '%.3f' % test_score["accuracy"],
                '%.3f' % test_score["absolute_true"],
                '%.3f' % test_score["absolute_false"],
                '%.3f' % runtime,
                now]]

    path = "{}/{}.csv".format('result', 'teacher')

    if os.path.exists(path):
        data1 = pd.read_csv(path, header=None)
        one_line = list(data1.iloc[0])
        if one_line == title:
            with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
                writer = csv.writer(t)  # 这一步是创建一个csv的写入器
                writer.writerows(content)  # 写入样本数据
        else:
            with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
                writer = csv.writer(t)  # 这一步是创建一个csv的写入器
                writer.writerow(title)  # 写入标签
                writer.writerows(content)  # 写入样本数据
    else:
        with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
            writer = csv.writer(t)  # 这一步是创建一个csv的写入器

            writer.writerow(title)  # 写入标签
            writer.writerows(content)  # 写入样本数据
    "---------------------------------------------------------------------------------------------------"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clip_pos = 0.7
    clip_neg = 0.5
    pos_weight = 0.3

    batch_size = 192
    epochs = 400
    # epochs = 15
    # learning_rate = 0.0018
    learning_rate = 0.018

    embedding_size = 192
    dropout = 0.6
    fan_epochs = 1
    num_heads = 8
    para = {'clip_pos': clip_pos,
            'clip_neg': clip_neg,
            'pos_weight': pos_weight,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'embedding_size': embedding_size,
            'dropout': dropout,
            'fan_epochs': fan_epochs,
            'num_heads': num_heads}
    # 可以选择main函数执行多少遍
    # for i in range(5):
    #     main(i, para)

    main(1, para)