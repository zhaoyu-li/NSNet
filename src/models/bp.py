import torch
import torch.nn as nn

from torch_scatter import scatter_sum, scatter_logsumexp


class BP(nn.Module):
    def __init__(self, opts):
        super(BP, self).__init__()
        self.opts = opts
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, data):
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        num_edges = data.num_edges

        sign_l_edge_index = data.sign_l_edge_index
        c2l_msg_repeat_index = data.c2l_msg_repeat_index
        c2l_msg_scatter_index = data.c2l_msg_scatter_index

        l2c_msg_aggr_repeat_index = data.l2c_msg_aggr_repeat_index
        l2c_msg_aggr_scatter_index = data.l2c_msg_aggr_scatter_index
        l2c_msg_scatter_index = data.l2c_msg_scatter_index
        
        if self.opts.task == 'model-counting':
            c_blf_repeat_index = data.c_blf_repeat_index
            c_blf_scatter_index = data.c_blf_scatter_index
            c_blf_norm_index = data.c_blf_norm_index
            v_degrees = data.v_degrees
            c_batch = data.c_batch
            v_batch = data.v_batch

        c2l_edges_feat = torch.zeros(num_edges, 1).to(self.opts.device)
        l2c_edges_feat = torch.zeros(num_edges, 1).to(self.opts.device)

        for _ in range(self.opts.n_rounds):
            c2l_msg = scatter_sum(c2l_edges_feat[c2l_msg_repeat_index], c2l_msg_scatter_index, dim=0, dim_size=num_edges)
            c2v_msg = c2l_msg.reshape(num_edges // 2, -1)
            norm = torch.logsumexp(c2v_msg, dim=1, keepdim=True)
            l2c_edges_feat = (c2v_msg - norm).reshape(num_edges, -1)
            
            l2c_msg_aggr = scatter_sum(l2c_edges_feat[l2c_msg_aggr_repeat_index], l2c_msg_aggr_scatter_index, dim=0, dim_size=l2c_msg_scatter_index.shape[0])
            c2l_edges_feat = scatter_logsumexp(l2c_msg_aggr, l2c_msg_scatter_index, dim=0, dim_size=num_edges)
        
        if self.opts.task == 'model-counting':
            c_blf_aggr = scatter_sum(l2c_edges_feat[c_blf_repeat_index], c_blf_scatter_index, dim=0, dim_size=c_blf_norm_index.shape[0])
            c_blf_norm = scatter_logsumexp(c_blf_aggr, c_blf_norm_index, dim=0, dim_size=c_size)
            c_norm_blf = c_blf_aggr - c_blf_norm[c_blf_norm_index]
            c_norm_blf[c_norm_blf==-float('inf')] = 0
            c_bethe = -scatter_sum(c_norm_blf * c_norm_blf.exp(), c_blf_norm_index, dim=0, dim_size=c_size).reshape(-1)

            l_blf_aggr = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            v_blf_aggr = l_blf_aggr.reshape(-1, 2)
            v_blf_norm = torch.logsumexp(v_blf_aggr, dim=1, keepdim=True)
            v_norm_blf = v_blf_aggr - v_blf_norm
            v_norm_blf[v_norm_blf==-float('inf')] = 0
            v_bethe = (v_degrees - 1) * ((v_norm_blf * v_norm_blf.exp()).sum(dim=1))

            return scatter_sum(c_bethe, c_batch, dim=0, dim_size=data.l_size.shape[0]) + \
                scatter_sum(v_bethe, v_batch, dim=0, dim_size=data.l_size.shape[0])
        else:
            l_logit = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            v_logit = l_logit.reshape(-1, 2)
            return self.softmax(v_logit)
