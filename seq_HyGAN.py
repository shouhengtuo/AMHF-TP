import warnings
warnings.filterwarnings('ignore')
from HyGmodel import *
import numpy as np # linear algebra
import dgl
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from get_Gf import gen_gf
# from util_metric import caculate_metric

def get_hyG(raw_path):
# def get_hyG(label_path, hyg_path, nodes_num):
    '''
    converting to sparse matrix
    '''
    label, nodes_num, subseq_idx_tensor, seq_idxs_tensor = gen_gf(raw_path)

    # label_data=pd.read_csv(label_path, header=None)
    # label=label_data[0].tolist()

    # LEN=4936
    LEN=9792
    nl=coo_matrix((LEN, LEN))
    nl.setdiag(1)
    values = nl.data
    indices = np.vstack((nl.row, nl.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = nl.shape
    nl=torch.sparse_coo_tensor(i, v, torch.Size(shape))
    nl=np.eye(LEN)
    nl=torch.from_numpy(nl)

    # chemicalsub_drug = torch.load(hyg_path)
    # data_dict = {
    #     ('node', 'in', 'edge'): (chemicalsub_drug[:, 0], chemicalsub_drug[:, 1]),
    #     ('edge', 'con', 'node'): (chemicalsub_drug[:, 1], chemicalsub_drug[:, 0])
    # }


    data_dict = {
            ('node', 'in', 'edge'): (subseq_idx_tensor, seq_idxs_tensor),
            ('edge', 'con', 'node'): (seq_idxs_tensor, subseq_idx_tensor)
        }



    '''
    finally passing the data_dict to construct hyG
    '''
    hyG = dgl.heterograph(data_dict)
    # 表示节点数量
    n_chemicalsub=nodes_num  #change the num of rows based on the dataset. row info. is available on data folder
    rows=n_chemicalsub
    n_hedge=LEN
    columns=n_hedge

    '''
    laoding the feature (one hot coding) for sequences (edges)
    '''
    # 边初始化 rows
    sequence_X=nl
    # 节点初始化，采用稀疏张量
    v_feat=coo_matrix((rows, 128))
    v_feat.setdiag(1)
    values = v_feat.data
    indices = np.vstack((v_feat.row, v_feat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = v_feat.shape
    v_feat=torch.sparse_coo_tensor(i, v, torch.Size(shape))

    hyG.ndata['h'] = {'edge' : sequence_X.type('torch.FloatTensor'), 'node' : v_feat.type('torch.FloatTensor')}
    e_feat = sequence_X.type('torch.FloatTensor')
    v_feat=v_feat.type('torch.FloatTensor')

    return sequence_X, dict(label=label, hyG=hyG, v_feat=v_feat, e_feat=e_feat)


# LEN, label, sequence_X, hyG, v_feat, e_feat = get_hyG("./ACPs_data/ACP2_alternate_train.tsv")
# # LEN, label, sequence_X, hyG, v_feat, e_feat = get_hyG("./data/label_cancer.txt",
# #                                                       "./data/hyG_cancer_kmer.pt", 10301)
#
# # print(hyG)
#
# num_class=len(set(label))
# test_size=0.1
# train_label, test_label= train_test_split(label, test_size=test_size, random_state=42)
# train_label=torch.tensor(train_label,dtype=torch.long)
#
# size=int((LEN)*0.1)
# val_label=train_label[0:size]
# train_label=train_label[size:]
# test_label=torch.tensor(test_label,dtype=torch.long)
#
# leanring_model = Seq_HyGAN(sequence_X.shape[1], 64, 128, 128, num_class, 0.3)
#
# def SeqHyGAN_train():
#
#     loss_fn = nn.CrossEntropyLoss()
#
#     optimizer = torch.optim.Adam(leanring_model.parameters(), lr=0.001)
#     best_val_acc = 0
#     patience = 0
#
#     for i in range(500):
#         leanring_model.train()
#         # 一次性把全部数据传入进去构造图，并输出全部的预测结果
#         pred = leanring_model(hyG, v_feat, e_feat, True, True)
#
#         train_pred, test_pred = train_test_split(pred, test_size=test_size, random_state=42)
#         val_pred = train_pred[0:size]
#         train_pred = train_pred[size:]
#
#         loss = loss_fn(train_pred, train_label)
#         pred_cls = torch.argmax(train_pred, -1)
#         train_acc = torch.eq(pred_cls, train_label).sum().item() / len(train_label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             leanring_model.eval()
#             val_cls = torch.argmax(val_pred, -1)
#             val_acc = torch.eq(val_cls, val_label).sum().item() / len(val_label)
#
#             # Save the best validation accuracy and the corresponding test accuracy.
#             if best_val_acc < val_acc:
#                 best_val_acc = val_acc
#                 patience = 0
#                 torch.save(leanring_model.state_dict(), './results/main0529.pkl')
#             else:
#                 patience += 1
#
#         if patience == 40:
#             break
#
#         if i % 10 == 0:
#             print('In epoch {}, train loss: {:.4f}, val_acc: {:.4f} (best_val_acc: {:.4f})'.format(i, loss, val_acc,
#                                                                                                    best_val_acc))
#
# label=torch.tensor(label,dtype=torch.long)
#
# def SeqHyGAN_test():
#     # 将保存验证结果最好的模型进行加载
#     leanring_model.load_state_dict(torch.load('./results/main0529.pkl'))
#     leanring_model.eval()
#     with torch.no_grad():
#         # 将构建的图数据传进去，获得的就是最好的预测结果
#         best_test_perd = leanring_model(hyG, v_feat, e_feat, True, True)
#         # 为了每次运行测试结果相同，不再每次随机划分测试集，将全部数据当作测试数据
#         # train_pred, test_pred = train_test_split(best_test_perd, test_size=test_size, random_state=42)
#         #
#         test_cls = torch.argmax(best_test_perd, -1)
#         pred_prob_positive = best_test_perd[:, 1]
#         metric, roc_data, prc_data= caculate_metric(test_cls, label,pred_prob_positive)
#         print('[ACC,\tSensitivity,\tSpecificity,\tAUC,\tMCC]')
#         print(metric.numpy())
#         # test_acc = torch.eq(test_cls, test_label).sum().item() / len(test_label)
#         # print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, auc:{:.4f}, MCC:{:.4f}'.format(
#         #         accuracy_score(
#         #             test_label.cpu(),
#         #             test_cls.cpu(),),
#         #         precision_score(
#         #             test_label.cpu(),
#         #             test_cls.cpu(),
#         #             average='weighted'),
#         #         recall_score(
#         #             test_label.cpu(),
#         #             test_cls.cpu(),
#         #             average='weighted'),
#         #         roc_auc_score(
#         #             test_label.cpu(),
#         #             test_cls.cpu(),average='weighted'),
#         #         matthews_corrcoef(
#         #             test_label.cpu(),
#         #             test_cls.cpu())))
#
#
# if __name__ == '__main__':
#     SeqHyGAN_train()
#     # SeqHyGAN_test()