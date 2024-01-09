# # import numpy as np
# #
# # # # 读取文件中的数据
# # # with open('hyG_fea.txt', 'r') as file:
# # #     train_lines = file.readlines()
# # #
# # #     x_train = []
# # #     i = 0
# # #     for line in train_lines:
# # #         # line_data1 = extract_numbers(line)
# # #         # x_train1 = np.concatenate([x_train0[i], line_data1])
# # #         # i += 1
# # #         # x_train.append(x_train1)
# # #         x_train1 = line
# # #     x_train = np.array(x_train)
# #
# #
# #
# # # 从TXT文件中读取数据
# # loaded_array = np.loadtxt('hyG_fea.txt')
# #
# # # 假设你的数据是二维的，这将把每一行作为一个一维数组
# # for row in loaded_array:
# #     print(row[1])
# #
# # def PadEncode(data, label, max_len):
# #     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
# #     data_e, label_e, seq_length, temp = [], [], [], []
# #     sign, b = 0, 0
# #     for i in range(len(data)):
# #         length = len(data[i])
# #         elemt, st = [], data[i].strip()
# #         for j in st:
# #             if j not in amino_acids:
# #                 sign = 1
# #                 break
# #             index = amino_acids.index(j)
# #             elemt.append(index)
# #             sign = 0
# #
# #         if length <= max_len and sign == 0:
# #             temp.append(elemt)
# #             seq_length.append(len(temp[b]))
# #             b += 1
# #             elemt += [0] * (max_len - length)
# #             data_e.append(elemt)
# #             label_e.append(label[i])
# #     return np.array(data_e), np.array(label_e), np.array(seq_length)
# #
# #
# # train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
# # y_train = np.array(train_sequence_label)
# #
# # x_train0, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
#
#
# valid_amino_acids = set('XACDEFGHIKLMNPQRSTVWY')
# data = []
# labels = []
#
# with open('dataset/target.txt', 'r') as file:
#     lines = file.readlines()
#
#     for i in range(0, len(lines), 2):
#         label = lines[i].strip()[1:]
#         data_line = lines[i + 1].strip()
#
#         # 检查数据行中是否所有字符都在有效氨基酸列表中
#         if all(char in valid_amino_acids for char in data_line):
#             labels.append(label)
#             data.append(data_line)
#         else:
#             print(f"Skipped data with label {label} due to invalid characters.")
#
# # 现在labels和data只包含有效的条目
# print("Valid Labels:", labels)
# print("Valid Data:", data)
import random

# 指定文件名
file_name = "random_numbers.txt"
# 指定要生成的行数
num_rows = 9792

# 生成随机数并写入文件
with open(file_name, "w") as file:
    for _ in range(num_rows):
        # 生成一行包含128个随机浮点数的数据，保留小数点后6位
        row = " ".join(f"{random.uniform(-10, 10):.6f}" for _ in range(128))
        file.write(row + "\n")

print(f"已生成文件: {file_name}")
