import os
import numpy as np
import h5py
from utils import ensure_dir
from config import ConfigLGAN
from trainer import TrainerLatentWGAN
from dataset.lgan_dataset import get_dataloader


cfg = ConfigLGAN()
print("data path:", cfg.data_root)
agent = TrainerLatentWGAN(cfg)

if not cfg.test:
    # load from checkpoint if provided
    if cfg.cont:
        agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader(cfg)

    agent.train(train_loader)
else:
    # load trained weights
    agent.load_ckpt(cfg.ckpt)

    # run generator
    generated_shape_codes = agent.generate(cfg.n_samples)

    # save generated z
    save_path = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}.h5".format(cfg.ckpt, cfg.n_samples))

    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("zs", shape=generated_shape_codes.shape, data=generated_shape_codes)
