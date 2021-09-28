import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import h5py
import shutil
import json
import random
import sys
sys.path.append("..")
from trainer.base import BaseTrainer
from utils import cycle, ensure_dirs, ensure_dir, read_ply, write_ply
try:
    from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
except Exception as e:
    print("need to install https://github.com/erikwijmans/Pointnet2_PyTorch")
    exit()


class Config(object):
    n_points = 2048
    batch_size = 128
    num_workers = 4
    nr_epochs = 200
    lr = 1e-4
    lr_step_size = 50
    # beta1 = 0.5
    grad_clip = None

    save_frequency = 100
    val_frequency = 10

    def __init__(self, args):
        self.data_root = os.path.join(args.proj_dir, args.exp_name, "results/all_zs_ckpt{}.h5".format(args.ae_ckpt))
        self.pc_root = args.pc_root
        self.split_path = args.split_path
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "pc2cad")
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.gpu_ids = args.gpu_ids

        if (not args.test) and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        if not args.test:
            os.system("cp pc2cad.py {}".format(self.exp_dir))
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(self.__dict__, f, indent=2)


class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()

        self.use_xyz = True

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[0, 32, 32, 64],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[64, 64, 64, 128],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                # bn=False,
                use_xyz=self.use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.Tanh()
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))


class TrainAgent(BaseTrainer):
    def build_net(self, config):
        self.net = PointNet2().cuda()

    def set_loss_function(self):
        self.criterion = nn.MSELoss().cuda()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr) # , betas=(config.beta1, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)

    def forward(self, data):
        points = data["points"].cuda()
        code = data["code"].cuda()

        pred_code = self.net(points)

        loss = self.criterion(pred_code, code)
        return pred_code, {"mse": loss}


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.data_root = config.data_root
        self.pc_root = config.pc_root
        self.path = config.split_path
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with h5py.File(self.data_root, 'r') as fp:
            self.zs = fp["{}_zs".format(phase)][:]

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + '.ply')
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        pc = read_ply(pc_path)
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]
        pc = torch.tensor(pc, dtype=torch.float32)
        shape_code = torch.tensor(self.zs[index], dtype=torch.float32)
        return {"points": pc, "code": shape_code, "id": data_id}

    def __len__(self):
        return len(self.zs)


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = ShapeCodesDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers)
    return dataloader


parser = argparse.ArgumentParser()
parser.add_argument('--proj_dir', type=str, default="proj_log",
                   help="path to project folder where models and logs will be saved")
parser.add_argument('--pc_root', type=str, default="path_to_pc_data", help="path to point clouds data folder")
parser.add_argument('--split_path', type=str, default="data/train_val_test_split.json", help="path to train-val-test split")
parser.add_argument('--exp_name', type=str, required=True, help="name of this experiment")
parser.add_argument('--ae_ckpt', type=str, required=True, help="desired checkpoint to restore")
parser.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
parser.add_argument('--test',action='store_true', help="test mode")
parser.add_argument('--n_samples', type=int, default=100, help="number of samples to generate when testing")
parser.add_argument('-g', '--gpu_ids', type=str, default="0",
                   help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
args = parser.parse_args()

if args.gpu_ids is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

cfg = Config(args)
print("data path:", cfg.data_root)
agent = TrainAgent(cfg)

if not args.test:
    # load from checkpoint if provided
    if args.cont:
        agent.load_ckpt(args.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    # start training
    clock = agent.clock

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = agent.train_func(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix({k: v.item() for k, v in losses.items()})

            # validation step
            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                outputs, losses = agent.val_func(data)

            clock.tick()

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            agent.save_ckpt()

        # if clock.epoch % 10 == 0:
        agent.save_ckpt('latest')
else:
    # load trained weights
    agent.load_ckpt(args.ckpt)

    test_loader = get_dataloader('test', cfg)

    save_dir = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}_pc".format(args.ckpt, args.n_samples))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_zs = []
    pbar = tqdm(test_loader)
    cnt = 0
    for i, data in enumerate(pbar):
        with torch.no_grad():
            pred_z, _ = agent.forward(data)
            pred_z = pred_z.detach().cpu().numpy()
            # print(pred_z.shape)
            all_zs.append(pred_z)
        pts = data['points'].detach().cpu().numpy()
        for j in range(pred_z.shape[0]):
            save_path = os.path.join(save_dir, "{}.ply".format(data['id'][j]))
            write_ply(pts[j], save_path)
        cnt += pred_z.shape[0]
        if cnt > args.n_samples:
            break

    all_zs = np.concatenate(all_zs, axis=0)
    # save generated z
    save_path = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}.h5".format(args.ckpt, args.n_samples))
    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("zs", shape=all_zs.shape, data=all_zs)
