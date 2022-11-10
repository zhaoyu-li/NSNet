import torch
import torch.nn as nn
import math

from models.mlp import MLP
from models.ln_lstm_cell import LayerNormBasicLSTMCell
from torch_scatter import scatter_sum, scatter_mean


class NeuroSAT(nn.Module):
    def __init__(self, opts):
        super(NeuroSAT, self).__init__()
        self.opts = opts
        self.l_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.c_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        self.l_update = LayerNormBasicLSTMCell(self.opts.dim * 2, self.opts.dim)
        self.denom = math.sqrt(self.opts.dim)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        num_edges = data.num_edges

        c_edge_index = data.c_edge_index
        l_edge_index = data.l_edge_index

        if self.opts.task == 'model-counting':
            l_batch = data.l_batch

        l_hidden = (self.l_init / self.denom).repeat(l_size, 1)
        c_hidden = (self.c_init / self.denom).repeat(c_size, 1)
        l_state = torch.zeros(l_size, self.opts.dim).to(self.opts.device)
        c_state = torch.zeros(c_size, self.opts.dim).to(self.opts.device)

        for _ in range(self.opts.n_rounds):
            l_msg_feat = self.l2c_msg_func(l_hidden)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            c_hidden, c_state = self.c_update(l2c_msg_aggr, (c_hidden, c_state))

            c_msg_feat = self.c2l_msg_func(c_hidden)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            pl_hidden, ul_hidden = torch.chunk(l_hidden.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_hidden, pl_hidden], dim=1).reshape(l_size, -1)
            l_hidden, l_state = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg], dim=1), (l_hidden, l_state))

        if self.opts.task == 'model-counting':
            l_logits = self.l_readout(l_hidden).reshape(-1)
            return scatter_mean(l_logits, l_batch, dim=0, dim_size=data.l_size.shape[0])
        else:
            l_logit = self.l_readout(l_hidden)
            v_logit = l_logit.reshape(-1, 2)
            return self.softmax(v_logit)
