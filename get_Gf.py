import random

import pandas as pd
import torch
import numpy as np


# # 将原文件正负样本进行随机打乱，并取得打乱后的标签数据，保存标签数据，返回序列数据
# def sf_line(raw_path):
#     # 读取原始文件
#     data0 = pd.read_csv(raw_path, sep='\t', header=0)
#     data = np.array(data0)
#     # 随机打乱行的顺序
#     random.shuffle(data)
#     # 保存所有序列数据
#     lables = []
#     seqs = []
#     for i in range(len(data)):
#         lables.append(data[:, 1][i])
#         seqs.append(data[:, 2][i])
#     return lables, seqs

def sf_line(raw_path):
    # getting sequence data and label
    data, label = [], []
    # path = "{}/{}.txt".format(first_dir, file_name)
    with open(raw_path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    # b保证数据的batchsize的整数倍 9792
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        st = data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            sign = 0
        if sign == 0:
            # seq_length.append(len(temp[b]))
            b += 1
            data_e.append(data[i])
            label_e.append(label[i])
            if b == 9792:
                break
    # return list(label), data
    return label_e, data_e


# 构建不重复的子序列词典
def build_subseq_dict(sequences):
    subseq_set = set()
    for seq in sequences:
        for i in range(len(seq) - 4):
            subseq_set.add(seq[i:i+5])
    subseq_dict = {subseq: idx for idx, subseq in enumerate(sorted(subseq_set))}
    return subseq_dict


def seqs_to_tensor(sequences, subseq_dict):
    subseq_idx = []
    seq_idxs = []
    for seq_idx, seq in enumerate(sequences):
        for i in range(len(seq) - 4):
            subseq = seq[i:i+5]
            subseq_idx.append(subseq_dict.get(subseq))
            seq_idxs.append(seq_idx)
    subseq_idx_tensor = torch.tensor(subseq_idx)
    seq_idxs_tensor = torch.tensor(seq_idxs)
    return subseq_idx_tensor, seq_idxs_tensor


def gen_gf(raw_path):
    lables, sequences = sf_line(raw_path)
    subseq_dict = build_subseq_dict(sequences)
    subseq_idx_tensor, seq_idxs_tensor = seqs_to_tensor(sequences, subseq_dict)
    nodes_num = len(subseq_dict)
    return lables, nodes_num, subseq_idx_tensor, seq_idxs_tensor


if __name__ == '__main__':
    lables, subseq_idx_tensor, seq_idxs_tensor = gen_gf("./ACPs_data/ACP2_alternate_train.tsv")
    print(seq_idxs_tensor)
# lan = torch.load('./idx_pairs.pt')
# for i in range(len(lan)):
#     print(lan[i])
# # print(np.array(lan).shape)

