import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py


def get_dataloader(cfg):
    dataset = LGANDataset(cfg.data_root)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=np.random.seed(), drop_last=True)
    return dataloader


class LGANDataset(Dataset):
    def __init__(self, data_root):
        super(LGANDataset, self).__init__()
        self.data_root = data_root
        with h5py.File(self.data_root, 'r') as fp:
            self.data = fp["train_zs"][:]

    def __getitem__(self, index):
        shape_code = torch.tensor(self.data[index], dtype=torch.float32)
        return shape_code

    def __len__(self):
        return len(self.data)
