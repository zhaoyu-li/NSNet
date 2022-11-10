import os
import glob
import torch
import pickle
import itertools
import numpy as np

from torch_geometric.data import Dataset
from torch_geometric.data import Data
from utils.utils import literal2l_idx, parse_cnf_file


class LCG(Data):
    def __init__(self, 
            l_size=None, 
            c_size=None, 
            c_edge_index=None, 
            l_edge_index=None,
            l_batch=None,
            c_batch=None
        ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.c_edge_index = c_edge_index
        self.l_edge_index = l_edge_index
        self.l_batch = l_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.c_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'c_edge_index':
            return self.c_size
        elif key == 'l_edge_index':
            return self.l_size
        elif key == 'l_batch' or key == 'c_batch':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class BPG(Data):
    def __init__(self, 
            l_size=None,
            c_size=None,
            sign_l_edge_index=None,
            c2l_msg_repeat_index=None,
            c2l_msg_scatter_index=None,
            l2c_msg_aggr_repeat_index=None,
            l2c_msg_aggr_scatter_index=None,
            l2c_msg_scatter_index=None,
            c_blf_repeat_index=None,
            c_blf_scatter_index=None,
            c_blf_norm_index=None,
            v_degrees=None,
            c_batch=None,
            v_batch=None,
            l_edge_index=None,
            c_edge_index=None
        ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.sign_l_edge_index = sign_l_edge_index
        self.c2l_msg_repeat_index = c2l_msg_repeat_index
        self.c2l_msg_scatter_index = c2l_msg_scatter_index
        self.l2c_msg_aggr_repeat_index = l2c_msg_aggr_repeat_index
        self.l2c_msg_aggr_scatter_index = l2c_msg_aggr_scatter_index
        self.l2c_msg_scatter_index = l2c_msg_scatter_index
        self.c_blf_repeat_index = c_blf_repeat_index
        self.c_blf_scatter_index = c_blf_scatter_index
        self.c_blf_norm_index = c_blf_norm_index
        self.v_degrees = v_degrees
        self.c_batch = c_batch
        self.v_batch = v_batch
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index
        
    @property
    def num_edges(self):
        return self.sign_l_edge_index.size(0)
       
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'c_blf_norm_index' or key == 'c_edge_index':
            return self.c_size
        elif key == 'sign_l_edge_index' or key == 'l_edge_index':
            return self.l_size
        elif key == 'c2l_msg_repeat_index' or key == 'c2l_msg_scatter_index' or key == 'l2c_msg_aggr_repeat_index' \
            or key == 'l2c_msg_scatter_index' or key == 'c_blf_repeat_index':
            return self.sign_l_edge_index.size(0)
        elif key == 'l2c_msg_aggr_scatter_index':
            return self.l2c_msg_scatter_index.size(0)
        elif key == 'c_blf_scatter_index':
            return self.c_blf_norm_index.size(0)
        elif key == 'c_batch' or key == 'v_batch':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class SATDataset(Dataset):
    def __init__(self, data_dir, data_size, opts):
        self.opts = opts
        all_files = sorted(glob.glob(data_dir + '/**/*.cnf', recursive=True))
        all_labels = self._get_labels(data_dir)
        if all_labels is None:
            all_labels = [None] * len(all_files)
        assert len(all_labels) == len(all_files)
        
        if data_size is not None:
            assert data_size <= len(all_files)
            self.file_indices = np.random.RandomState(0).permutation(len(all_files))[:data_size]
            self.all_files = all_files[self.file_indices]
            self.all_labels = all_labels[self.file_indices]
        else:
            self.file_indices = list(range(len(all_files)))
            self.all_files = all_files
            self.all_labels = all_labels
        
        if self.opts.model == 'NeuroSAT':
            self.graph = 'LCG'
        else:
            self.graph = 'BPG'
                    
        super().__init__(data_dir)
    
    @property
    def processed_file_names(self):
        return [f'data_{idx}_{self.graph}_{self.opts.task}.pt' for idx in self.file_indices]
    
    def process(self):
        idx = 0
        for file_path, label in zip(self.all_files, self.all_labels):
            n_vars, clauses = parse_cnf_file(file_path)
            if self.graph == 'LCG':
                data = self._transform2LCG(n_vars, clauses)
            else:
                data = self._transform2BPG(n_vars, clauses)
            file_name = f'data_{idx}_{self.graph}_{self.opts.task}.pt'
            torch.save(data, os.path.join(self.processed_dir, file_name))
            idx += 1

    def _transform2LCG(self, n_vars, clauses):
        c_edge_index_list = []
        l_edge_index_list = []

        l_batch = None
        c_batch = None

        if self.opts.task == 'model-counting':
            l_batch = torch.zeros(n_vars * 2, dtype=torch.long)
        else:
            c_batch = torch.zeros(len(clauses), dtype=torch.long)
        
        for c_idx, clause in enumerate(clauses):
            for literal in clause:
                l_idx = literal2l_idx(literal)
                c_edge_index_list.append(c_idx)
                l_edge_index_list.append(l_idx)
        
        c_edge_index = torch.tensor(c_edge_index_list, dtype=torch.long)
        l_edge_index = torch.tensor(l_edge_index_list, dtype=torch.long)

        return LCG(
            n_vars * 2,
            len(clauses),
            c_edge_index,
            l_edge_index,
            l_batch,
            c_batch
        )
    
    def _transform2BPG(self, n_vars, clauses):
        sign_l_edge_index_list = []
        type_edge_index_list = []

        c2l_msg_aggr_c_index_map = {l: [] for l in range(2 * n_vars)}
        c2l_msg_aggr_edge_index_map = {l: [] for l in range(2 * n_vars)}
        
        c2l_msg_repeat_index_list = []
        c2l_msg_scatter_index_list = []

        l2c_msg_aggr_repeat_index_list = []
        l2c_msg_aggr_scatter_index_list = []
        l2c_msg_scatter_index_list = []

        # auxiliary parameters
        c_blf_repeat_index = None
        c_blf_scatter_index = None
        c_blf_norm_index = None
        v_degrees = None
        c_batch = None
        v_batch = None
        l_edge_index = None
        c_edge_index = None

        if self.opts.task == 'model-counting':
            c_blf_repeat_index_list = []
            c_blf_scatter_index_list = []
            c_blf_norm_index_list = []
            v_degrees = torch.zeros(n_vars)
        else:
            l_edge_index_list = []
            c_edge_index_list = []
        
        index_base = 0
        msg_aggr_index = 0
        for c_idx, clause in enumerate(clauses):
            used_vars = sorted(list(set([abs(literal)-1 for literal in clause])))
            # literal to clause message
            for msg_idx, v_idx in enumerate(used_vars):
                pl_idx = v_idx * 2
                nl_idx = v_idx * 2 + 1
                p_msg_idx = index_base + msg_idx * 2
                n_msg_idx = index_base + msg_idx * 2 + 1
                
                sign_l_edge_index_list.append(pl_idx)
                sign_l_edge_index_list.append(nl_idx)
                
                c2l_msg_aggr_c_index_map[pl_idx].append(c_idx)
                c2l_msg_aggr_c_index_map[nl_idx].append(c_idx)
                
                c2l_msg_aggr_edge_index_map[pl_idx].append(p_msg_idx)
                c2l_msg_aggr_edge_index_map[nl_idx].append(n_msg_idx)
                
                if (v_idx + 1) in clause:
                    type_edge_index_list.append(1)
                else:
                    type_edge_index_list.append(0)
                
                if -(v_idx + 1) in clause:
                    type_edge_index_list.append(1)
                else:
                    type_edge_index_list.append(0)
            
            # clause to literal massage
            for scatter_msg_idx, discard_v_idx in enumerate(used_vars):
                for indices in np.ndindex(tuple([2] * (len(used_vars)-1))):
                    msg_table = [(index_base + msg_idx * 2 + 1, index_base + msg_idx * 2) 
                        for msg_idx, v_idx in enumerate(used_vars) if v_idx != discard_v_idx]
                    assign = np.array([type_edge_index_list[msg_table[i][idx]] for i, idx in enumerate(indices)])
                    is_sat = assign.sum() > 0
                    msg_aggr_repeat_index = [msg_table[i][idx] for i, idx in enumerate(indices)]
                    
                    if type_edge_index_list[index_base + scatter_msg_idx * 2] or is_sat:       
                        l2c_msg_aggr_repeat_index_list.append(msg_aggr_repeat_index)
                        l2c_msg_aggr_scatter_index_list.append([msg_aggr_index] * len(msg_aggr_repeat_index))
                        msg_aggr_index += 1
                        l2c_msg_scatter_index_list.append(index_base + scatter_msg_idx * 2)
                    
                    if type_edge_index_list[index_base + scatter_msg_idx * 2 + 1] or is_sat:
                        l2c_msg_aggr_repeat_index_list.append(msg_aggr_repeat_index)
                        l2c_msg_aggr_scatter_index_list.append([msg_aggr_index] * len(msg_aggr_repeat_index))
                        msg_aggr_index += 1
                        l2c_msg_scatter_index_list.append(index_base + scatter_msg_idx * 2 + 1)
            
            index_base += len(used_vars) * 2

        sign_l_edge_index = torch.tensor(sign_l_edge_index_list, dtype=torch.long)

        index_base = 0
        for c_idx, clause in enumerate(clauses):
            used_vars = sorted(list(set([abs(literal)-1 for literal in clause])))
            for msg_idx, v_idx in enumerate(used_vars):
                pl_idx = v_idx * 2
                nl_idx = v_idx * 2 + 1
                p_msg_idx = index_base + msg_idx * 2
                n_msg_idx = index_base + msg_idx * 2 + 1
                
                for neighbor_c_idx, neighbor_msg_idx in zip(c2l_msg_aggr_c_index_map[pl_idx], c2l_msg_aggr_edge_index_map[pl_idx]):
                    if neighbor_c_idx == c_idx:
                        continue
                    c2l_msg_repeat_index_list.append(neighbor_msg_idx)
                    c2l_msg_scatter_index_list.append(p_msg_idx)
                
                for neighbor_c_idx, neighbor_msg_idx in zip(c2l_msg_aggr_c_index_map[nl_idx], c2l_msg_aggr_edge_index_map[nl_idx]):
                    if neighbor_c_idx == c_idx:
                        continue
                    c2l_msg_repeat_index_list.append(neighbor_msg_idx)
                    c2l_msg_scatter_index_list.append(n_msg_idx)
            
            index_base += len(used_vars) * 2
        
        c2l_msg_repeat_index = torch.tensor(c2l_msg_repeat_index_list, dtype=torch.long)
        c2l_msg_scatter_index = torch.tensor(c2l_msg_scatter_index_list, dtype=torch.long)

        l2c_msg_aggr_repeat_index_list = list(itertools.chain(*l2c_msg_aggr_repeat_index_list))
        l2c_msg_aggr_scatter_index_list = list(itertools.chain(*l2c_msg_aggr_scatter_index_list))

        l2c_msg_aggr_repeat_index = torch.tensor(l2c_msg_aggr_repeat_index_list, dtype=torch.long)
        l2c_msg_aggr_scatter_index = torch.tensor(l2c_msg_aggr_scatter_index_list, dtype=torch.long)
        l2c_msg_scatter_index = torch.tensor(l2c_msg_scatter_index_list, dtype=torch.long)

        if self.opts.task == 'model-counting':
            index_base = 0
            blf_index = 0
            for c_idx, clause in enumerate(clauses):
                used_vars = set([abs(literal)-1 for literal in clause])
                for indices in np.ndindex(tuple([2] * len(used_vars))):
                    msg_table = [(index_base + msg_idx * 2 + 1, index_base + msg_idx * 2) 
                        for msg_idx, v_idx in enumerate(used_vars)]
                    assign = np.array([type_edge_index_list[msg_table[i][idx]] for i, idx in enumerate(indices)])
                    is_sat = assign.sum() > 0
                    blf_repeat_index = [msg_table[i][idx] for i, idx in enumerate(indices)]
                    
                    if is_sat:
                        c_blf_repeat_index_list.append(blf_repeat_index)
                        c_blf_scatter_index_list.append([blf_index] * len(blf_repeat_index))
                        blf_index += 1
                        c_blf_norm_index_list.append(c_idx)
                index_base += len(used_vars) * 2

                for v_idx in used_vars:
                    v_degrees[v_idx] += 1
            
            c_blf_repeat_index_list = list(itertools.chain(*c_blf_repeat_index_list))
            c_blf_scatter_index_list = list(itertools.chain(*c_blf_scatter_index_list))

            c_blf_repeat_index = torch.tensor(c_blf_repeat_index_list, dtype=torch.long)
            c_blf_scatter_index = torch.tensor(c_blf_scatter_index_list, dtype=torch.long)
            c_blf_norm_index = torch.tensor(c_blf_norm_index_list, dtype=torch.long)

            c_batch = torch.zeros(len(clauses), dtype=torch.long)
            v_batch = torch.zeros(n_vars, dtype=torch.long)
        else:
            for c_idx, clause in enumerate(clauses):
                for literal in clause:
                    l_idx = literal2l_idx(literal)
                    l_edge_index_list.append(l_idx)
                    c_edge_index_list.append(c_idx)
            
            l_edge_index = torch.tensor(l_edge_index_list, dtype=torch.long)
            c_edge_index = torch.tensor(c_edge_index_list, dtype=torch.long)
            c_batch = torch.zeros(len(clauses), dtype=torch.long)

        return BPG(
            n_vars*2, 
            len(clauses),
            sign_l_edge_index, 
            c2l_msg_repeat_index,
            c2l_msg_scatter_index,
            l2c_msg_aggr_repeat_index,
            l2c_msg_aggr_scatter_index,
            l2c_msg_scatter_index,
            c_blf_repeat_index,
            c_blf_scatter_index,
            c_blf_norm_index,
            v_degrees,
            c_batch,
            v_batch,
            l_edge_index,
            c_edge_index,
        )

    def _get_labels(self, data_dir):
        if self.opts.task == 'model-counting':
            labels = None
            labels_file = os.path.join(data_dir, 'countings.pkl')
            if os.path.exists(labels_file):
                with open(labels_file, 'rb') as f:
                    labels = pickle.load(f)
                labels = [torch.tensor(label, dtype=torch.float) for label in labels]
            return labels
        elif self.opts.task == 'sat-solving':
            labels = None
            if hasattr(self.opts, 'loss'):
                if self.opts.loss == 'assignment':
                    labels_file = os.path.join(data_dir, 'assignments.pkl')
                else:
                    assert self.opts.loss == 'marginal'
                    labels_file = os.path.join(data_dir, 'marginals.pkl')
                if os.path.exists(labels_file):
                    with open(labels_file, 'rb') as f:
                        labels = pickle.load(f)
                    labels = [torch.tensor(label, dtype=torch.float) for label in labels]
            return labels
    
    def len(self):
        return len(self.all_files)

    def get(self, idx):
        file_name = f'data_{idx}_{self.graph}_{self.opts.task}.pt'
        data = torch.load(os.path.join(self.processed_dir, file_name))
        data.y = self.all_labels[idx]
        return data
