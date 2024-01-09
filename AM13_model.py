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


        self.full3 = nn.Linear(149, 500)
        self.full4 = nn.Linear(500, 250)
        self.full5 = nn.Linear(250, 128)

        # self.full6 = nn.Linear(4608, 2304)
        # self.Flatten = nn.Linear(128, 64)
        self.Flatten = nn.Linear(149, 64)



        self.cls = nn.Linear(64, num_class)
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
            # [9792, 128]
            feat_e = hyG.ndata['h']['edge']
           
            
            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
            if last_layer:
                loaded_array = np.loadtxt('bert_fea.txt')
                # 将NumPy数组转换为PyTorch张量
                # [9843,21]
                features_tensor = torch.from_numpy(loaded_array).float()
                bert_fea = features_tensor[0:9792, :]
                # 拼接，21+128 = 149
                output0 = torch.cat([feat_e, bert_fea], dim=-1)

                label = self.full3(output0)
                label = torch.nn.ReLU()(label)
                label1 = self.full4(label)
                label = torch.nn.ReLU()(label1)
                label2 = self.full5(label)
                label = torch.nn.ReLU()(label2)
                label3 = self.Flatten(output0)
                label = torch.nn.ReLU()(label3)

                pred=self.cls(label) #to reduce the hyperedge feature into 'c' dimension, where c is the number of class
                return pred
            else:
                return hyG, feat_v, feat_e