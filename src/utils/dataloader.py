import torch
import numpy as np
import random

from utils.dataset import SATDataset
from torch_geometric.loader import DataLoader


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloader(data_dir, opts, mode, data_size=None):
    dataset = SATDataset(data_dir, data_size, opts)
    return DataLoader(
        dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=(mode=='train'),
        worker_init_fn=_worker_init_fn,
        pin_memory=True
    )
