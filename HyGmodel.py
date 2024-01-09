import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Seq_HyGAN(nn.Module):
    def __init__(self, i_d, q_d, v_d, e_d, num_class, dropout = 0.5):
        # q_d, v_d, e_d :64 128 128
        super(Seq_HyGAN, self).__init__()
        self.dropout = dropout
        
        self.q_d = q_d
        self.first_layer_in = torch.nn.Linear(i_d, v_d)
        self.not_first_layer_in = torch.nn.Linear(v_d, v_d)

        self.w1 = torch.nn.Linear(e_d, q_d)
        self.w2 = torch.nn.Linear(v_d, q_d)
        self.w3 = torch.nn.Linear(v_d, e_d)
        self.w4 = torch.nn.Linear(v_d, q_d)
        self.w5 = torch.nn.Linear(e_d, q_d)
        self.w6 = torch.nn.Linear(e_d, v_d)
        self.cls = nn.Linear(e_d, num_class)
    def red_function(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggreated = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggreated}
    def attention(self, edges):
        attention_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        c=attention_score/np.sqrt(self.q_d)
        return {'Attn': c}
    def msg_function(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']} 
  
    def forward(self, hyG, vfeat, efeat,first_layer,last_layer):
        with hyG.local_scope():
            if first_layer:
                feat_e = self.first_layer_in(efeat)
            else:
                feat_e = self.not_first_layer_in(efeat)                    
            feat_v = vfeat

            # Hyperedge attention
            hyG.ndata['h'] = {'edge': feat_e}
            hyG.ndata['k'] = {'edge' : self.w5(feat_e)}
            hyG.ndata['v'] = {'edge' : self.w6(feat_e)}
            hyG.ndata['q'] = {'node' : self.w4(feat_v)}
            hyG.apply_edges(self.attention, etype='con')
            hyG.update_all(self.msg_function, self.red_function, etype='con')
            feat_v = hyG.ndata['h']['node']

            # node attention
            hyG.ndata['k'] = {'node' : self.w2(feat_v)}
            hyG.ndata['v'] = {'node' : self.w3(feat_v)}
            hyG.ndata['q'] = {'edge' : self.w1(feat_e)}
            hyG.apply_edges(self.attention, etype='in')
            hyG.update_all(self.msg_function, self.red_function, etype='in')
            feat_e = hyG.ndata['h']['edge']
           
            
            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
            if last_layer:
                # 将图模型的输出特征保存到文件中，作为特征汇总到不同的模型进行训练
                # 将Tensor转换为NumPy数组
                numpy_array = feat_e.numpy()

                # 将NumPy数组保存为TXT文件
                np.savetxt('hyG_fea1125.txt', numpy_array, fmt='%f')
                # np.savetxt('hyG_fea.txt', numpy_array)

                pred=self.cls(feat_e) #to reduce the hyperedge feature into 'c' dimension, where c is the number of class
                return pred
            else:
                return hyG, feat_v, feat_e