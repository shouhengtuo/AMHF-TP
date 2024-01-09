#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 17:13
# @Author  : fhh
# @File    : train.py
# @Software: PyCharm
import time
import torch
import math
import numpy as np
from sklearn import metrics
# from torchinfo import summary
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
import evaluation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def hyG_evaluate(model, best_test_pred, test_label, device="cpu"):
    model.to(device)
    # model.load_state_dict(torch.load('./result/hyg01.pkl'))
    model.eval()
    # test_data['hyG'] = test_data['hyG'].to(device)
    # test_data['v_feat'] = test_data['v_feat'].to(device)
    # test_data['e_feat'] = test_data['e_feat'].to(device)
    with torch.no_grad():
        #     predictions = model(test_data['hyG'], test_data['v_feat'], test_data['e_feat'], True, True).tolist()
        #     labels = test_data['label']

        scores = evaluation.evaluate(np.array(best_test_pred), np.array(test_label))
    return scores


def evaluate(model, datadl, device="cpu"):
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y, z in datadl:
            x = x.to(device).long()
            y = y.to(device).long()
            predictions.extend(model(x, z).tolist())
            labels.extend(y.tolist())

    scores = evaluation.evaluate(np.array(predictions), np.array(labels))
    return scores

def scoring(y_true, y_score):
    threshold = 0.5
    y_pred = [int(i >= threshold) for i in y_score]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.flatten()
    sen = tp / (fn + tp)
    spe = tn / (fp + tn)
    pre = metrics.precision_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_score)
    pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(rc, pr)
    f1 = metrics.f1_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)


class DataTrain:
    # def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cpu", *args, **kwargs):
        self.model = model

        # self.hyG = hyG
        # self.v_feat = v_feat
        # self.e_feat = e_feat
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler

        self.device = device
        self.model.to(self.device)
        # self.hyG.to(self.device)
        # self.v_feat.to(self.device)
        # self.e_feat.to(self.device)

    # def train_step(self, train_iter, epochs=None, plot_picture=False):
    #     x_plot = []
    #     y_plot = []
    #     epochTrainLoss = []
    #     # for train_data, train_label in train_iter:
    #     #     train_data, train_label = train_data.to(self.device), train_label.to(self.device)
    #     #     summary(self.model, train_data.shape, dtypes=['torch.IntTensor'])
    #     #     break
    #     steps = 1
    #     for epoch in range(1, epochs+1):
    #         # 每次迭代将全部数据送入图网络中，？随着网络更新权重
    #         hyG_fea = self.model(self.hyG, self.v_feat, self.e_feat, True, True)
    #
    #         # metric = Accumulator(2)
    #         start = time.time()
    #         total_loss = 0
    #         i = 0
    #         batch_size = 192
    #         for train_data, train_label, train_length in train_iter:
    #             start_index = i * batch_size
    #             i += 1
    #             end_index = start_index + batch_size
    #             # 从数据集中切片获取当前批次数据
    #             batch_hyG_fea = hyG_fea[start_index:end_index]
    #             batch_hyG_fea = batch_hyG_fea.to(self.device)
    #             batch_hyG_fea.requires_grad_()
    #
    #             # self.model.train()  # 进入训练模式
    #             train_data, train_label, train_length = train_data.to(self.device), train_label.to(self.device), train_length.to(self.device)
    #             # y_hat = self.model(train_data.long(), train_length)
    #             # loss = self.criterion(y_hat, train_label.float())
    #             loss = self.criterion(batch_hyG_fea, train_label.float())
    #             self.optimizer.zero_grad()
    #             loss.backward(retain_graph=True)
    #             self.optimizer.step()
    #
    #             if self.lr_scheduler:
    #                 if self.lr_scheduler.__module__ == lr_scheduler.__name__:
    #                     # Using PyTorch In-Built scheduler
    #                     self.lr_scheduler.step()
    #                 else:
    #                     # Using custom defined scheduler
    #                     for param_group in self.optimizer.param_groups:
    #                         param_group['lr'] = self.lr_scheduler(steps)
    #
    #             x_plot.append(epoch)
    #             y_plot.append(self.lr_scheduler(epoch))
    #             total_loss += loss.item()
    #             steps += 1
    #         # train_loss = metric[0] / metric[1]
    #         # x_plot.append(self.lr_scheduler.last_epoch)
    #         # y_plot.append(self.lr_scheduler.get_lr()[0])
    #         # x_plot.append(epoch)
    #         # y_plot.append(self.lr_scheduler(epoch))
    #         # epochTrainLoss.append(train_loss)
    #         finish = time.time()
    #
    #         print(f'[ Epoch {epoch} ', end='')
    #         print("运行时间{}s".format(finish - start))
    #         print(f'loss={total_loss / len(train_iter)} ]')
    #
    #     if plot_picture:
    #         # 绘制学习率变化曲线
    #         plt.plot(x_plot, y_plot, 'r')
    #         plt.title('lr value of LambdaLR with (Cos_warmup) ')
    #         plt.xlabel('step')
    #         plt.ylabel('lr')
    #         plt.savefig('./result/Cos_warmup.jpg')
    #         # plt.show()
    #
    #         # # 绘制损失函数曲线
    #         # plt.figure()
    #         # plt.plot(x_plot, epochTrainLoss)
    #         # plt.legend(['trainLoss'])
    #         # plt.xlabel('epochs')
    #         # plt.ylabel('SHLoss')
    #         # plt.savefig('./image/Cos_warmup_Loss(200).jpg')
    #         # # plt.show()
    def train_step(self, train_data, epochs=None, plot_picture=False):
        x_plot = []
        y_plot = []
        y_train = torch.Tensor(train_data['label'])
        epochTrainLoss = []
        # for train_data, train_label in train_iter:
        #     train_data, train_label = train_data.to(self.device), train_label.to(self.device)
        #     summary(self.model, train_data.shape, dtypes=['torch.IntTensor'])
        #     break
        steps = 1
        # test_size = 0.1
        test_size = 0.2 # 使用这个划分形式，将测试训练数据分割方式与模型1基本相同，从而更好的合并数据
        LEN = len(y_train)
        train_label, test_label = train_test_split(y_train, test_size=test_size, random_state=42)

        size = int((LEN) * 0.1)
        val_label = train_label[0:size]
        train_label = train_label[size:]
        test_label = torch.tensor(test_label, dtype=torch.long)
        best_aiming = 0.4
        best_test_pred = torch.empty(0, 2)
        for epoch in range(1, epochs+1):
            self.model.train()  # 进入训练模式
            # 每次迭代将全部数据送入图网络中，随着网络更新权重
            # 手动选择模型迭代多少次
            # # 迭代3次
            # hyG, feat_v, feat_e = self.model(train_data['hyG'], train_data['v_feat'], train_data['e_feat'], True, False)
            # hyG, feat_v, feat_e = self.model(hyG, feat_v, feat_e, False, False)
            # hyG_fea = self.model(hyG, feat_v, feat_e, False, True)
            # 迭代1次
            # hyG_fea = self.model(train_data['hyG'], train_data['v_feat'], train_data['e_feat'], True, True)

            # 迭代2次
            hyG, feat_v, feat_e = self.model(train_data['hyG'], train_data['v_feat'], train_data['e_feat'], True, False)
            hyG_fea = self.model(hyG, feat_v, feat_e, False, True)

            # metric = Accumulator(2)
            start = time.time()
            total_loss = 0
            # 将训练好的数据再分成训练数据和验证数据，训练数据进行反向传播计算梯度，验证数据用来验证模型保存最好结果的模型
            train_pred, test_pred = train_test_split(hyG_fea, test_size=test_size, random_state=42)
            val_pred = train_pred[0:size]
            train_pred = train_pred[size:]
            loss = self.criterion(train_pred, train_label.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                    # Using PyTorch In-Built scheduler
                    self.lr_scheduler.step()
                else:
                    # Using custom defined scheduler
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_scheduler(steps)
            x_plot.append(epoch)
            y_plot.append(self.lr_scheduler(epoch))
            total_loss += loss.item()
            steps += 1
            # train_loss = metric[0] / metric[1]
            # x_plot.append(self.lr_scheduler.last_epoch)
            # y_plot.append(self.lr_scheduler.get_lr()[0])
            # x_plot.append(epoch)
            # y_plot.append(self.lr_scheduler(epoch))
            # epochTrainLoss.append(train_loss)
            finish = time.time()
            with torch.no_grad():
                # predictions = self.model(test_data['hyG'], test_data['v_feat'], test_data['e_feat'], True, True).tolist()
                # labels = test_data['label']
                val_scores = evaluation.evaluate(np.array(val_pred), np.array(val_label))
                val_aiming = val_scores["aiming"]
                if val_aiming > best_aiming:
                    best_aiming = val_aiming
                    best_test_pred = test_pred
                    torch.save(self.model.state_dict(), './result/hyg1117.pkl')

            print(f'[ Epoch {epoch}', end='')
            print(" 运行时间{}s".format(finish - start), end='')
            print(f' loss={loss}', end='')
            print(f' val_aiming={val_aiming} ]')
        return best_test_pred, test_label

    def load_model_test(self, model, train_data):
        model.load_state_dict(torch.load('./result/hyg1117.pkl'))
        with torch.no_grad():
            model.eval()
            test_size = 0.1
            y_train = torch.Tensor(train_data['label'])
            # 划分
            train_label, test_label = train_test_split(y_train, test_size=test_size, random_state=42)
            hyG_fea = model(train_data['hyG'], train_data['v_feat'], train_data['e_feat'], True, True)

            # # 将图模型的输出特征保存到文件中，作为特征汇总到不同的模型进行训练
            # # 将Tensor转换为NumPy数组
            # numpy_array = hyG_fea.numpy()
            #
            # # 将NumPy数组保存为TXT文件
            # # np.savetxt('hyG_fea.txt', numpy_array, fmt='%f')
            # np.savetxt('hyG_fea.txt', numpy_array, fmt='%f')

            train_pred, test_pred = train_test_split(hyG_fea, test_size=test_size, random_state=42)
        return test_pred, test_label

    def KD_step(self, train_iter, epochs=None, plot_picture=False):
        steps = 1
        for epoch in range(1, epochs+1):
            # metric = Accumulator(2)
            start = time.time()
            total_loss = 0
            for train_data, train_label in train_iter:
                self.model.train()  # 进入训练模式

                train_data, train_label = train_data.to(self.device), train_label.to(self.device)
                y_hat = self.model(train_data.long())
                loss = self.criterion(y_hat, train_label.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                # x_plot.append(epoch)
                # y_plot.append(self.lr_scheduler(epoch))
                total_loss += loss.item()
                steps += 1
            finish = time.time()

            print(f'[ Epoch {epoch} ', end='')
            print("运行时间{}s".format(finish - start))
            print(f'loss={total_loss / len(train_iter)} ]')


def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
