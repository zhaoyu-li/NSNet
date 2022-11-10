import torch
import torch.nn as nn
import math

from models.mlp import MLP
from torch_scatter import scatter_sum, scatter_logsumexp


class NSNet(nn.Module):
    def __init__(self, opts):
        super(NSNet, self).__init__()
        self.opts = opts
        self.c2l_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.denom = math.sqrt(self.opts.dim)
        self.c2l_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_norm = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
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
            c_bethes = []
            v_bethes = []
        
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(num_edges, 1)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(num_edges, 1)

        for _ in range(self.opts.n_rounds):
            c2l_msg = scatter_sum(c2l_edges_feat[c2l_msg_repeat_index], c2l_msg_scatter_index, dim=0, dim_size=num_edges)
            l2c_edges_feat_new = self.l2c_msg_update(c2l_msg)
            v2c_edges_feat_new = l2c_edges_feat_new.reshape(num_edges // 2, -1)
            pv2c_edges_feat_new, nv2c_edges_feat_new = torch.chunk(v2c_edges_feat_new, 2, 1)
            l2c_edges_feat_inv = torch.cat([nv2c_edges_feat_new, pv2c_edges_feat_new], dim=1).reshape(num_edges, -1)
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_edges_feat_new, l2c_edges_feat_inv], dim=1))

            l2c_msg_aggr = scatter_sum(l2c_edges_feat[l2c_msg_aggr_repeat_index], l2c_msg_aggr_scatter_index, dim=0, dim_size=l2c_msg_scatter_index.shape[0])
            l2c_msg = scatter_logsumexp(l2c_msg_aggr, l2c_msg_scatter_index, dim=0, dim_size=num_edges)
            c2l_edges_feat = self.c2l_msg_update(l2c_msg)

        if self.opts.task == 'model-counting':
            c_blf_aggr = scatter_sum(l2c_edges_feat[c_blf_repeat_index], c_blf_scatter_index, dim=0, dim_size=c_blf_norm_index.shape[0])
            c_blf_aggr = self.c_readout(c_blf_aggr)
            c_blf_norm = scatter_logsumexp(c_blf_aggr, c_blf_norm_index, dim=0, dim_size=c_size)
            c_norm_blf = c_blf_aggr - c_blf_norm[c_blf_norm_index]
            c_bethe = -scatter_sum(c_norm_blf * c_norm_blf.exp(), c_blf_norm_index, dim=0, dim_size=c_size).reshape(-1)

            l_blf_aggr = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            l_blf_aggr = self.l_readout(l_blf_aggr)
            v_blf_aggr = l_blf_aggr.reshape(-1, 2)
            v_blf_norm = torch.logsumexp(v_blf_aggr, dim=1, keepdim=True)
            v_norm_blf = v_blf_aggr - v_blf_norm
            v_bethe = (v_degrees - 1) * ((v_norm_blf * v_norm_blf.exp()).sum(dim=1))

            return scatter_sum(c_bethe, c_batch, dim=0, dim_size=data.l_size.shape[0]) + \
                scatter_sum(v_bethe, v_batch, dim=0, dim_size=data.l_size.shape[0])
        else:
            l_logit = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            l_logit = self.l_readout(l_logit)
            v_logit = l_logit.reshape(-1, 2)
            return self.softmax(v_logit)
