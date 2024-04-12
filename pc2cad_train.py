import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import TrainClock, cycle, ensure_dirs, ensure_dir
import argparse
import h5py
import shutil
import json
import random
from plyfile import PlyData, PlyElement
import sys
sys.path.append("..")
from agent import BaseAgent
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from plyfile import PlyData, PlyElement


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


class Config(object):
    n_points = 2048
    batch_size = 128
    num_workers = 8
    nr_epochs = 200
    lr = 1e-4
    lr_step_size = 50
    # beta1 = 0.5
    grad_clip = None
    noise = 0.02

    save_frequency = 100
    val_frequency = 10

    def __init__(self, args):
        self.data_root = os.path.join(args.proj_dir, args.exp_name, "results/all_zs_ckpt{}.h5".format(args.ae_ckpt))
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "pc2cad_tune_noise{}_{}_new".format(self.n_points, self.noise))
        print(self.exp_dir)
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


class EncoderPointNet(nn.Module):
    def __init__(self, n_filters=(128, 256, 512, 1024), bn=True):
        super(EncoderPointNet, self).__init__()
        self.n_filters = list(n_filters) #  + [latent_dim]
        # self.latent_dim = latent_dim

        model = []
        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = torch.mean(x, dim=2)
        x = self.fc_layer(x)
        return x


class TrainAgent(BaseAgent):
    def build_net(self, config):
        net = PointNet2()
        if len(config.gpu_ids) > 1:
            net = nn.DataParallel(net)
        # net = EncoderPointNet()
        return net

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


def read_ply(path, with_normal=False):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
        if with_normal:
            nx = np.array(plydata['vertex']['nx'])
            ny = np.array(plydata['vertex']['ny'])
            nz = np.array(plydata['vertex']['nz'])
            normals = np.stack([nx, ny, nz], axis=1)
    if with_normal:
        return np.concatenate([vertex, normals], axis=1)
    else:
        return vertex


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.data_root = config.data_root
        # self.abc_root = "/mnt/disk6/wurundi/abc"
        self.abc_root = "/home/rundi/data/abc"
        self.pc_root = self.abc_root + "/pc_v5a_processed_merge"
        self.path = os.path.join(self.abc_root, "cad_e10_l6_c15_len60_min0_t100.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with h5py.File(self.data_root, 'r') as fp:
            self.zs = fp["{}_zs".format(phase)][:]

        self.noise = config.noise

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + '.ply')
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        pc_n = read_ply(pc_path, with_normal=True)
        pc = pc_n[:, :3]
        normal = pc_n[:, -3:]
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]
        normal = normal[sample_idx]
        normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6)
        pc = pc + np.random.uniform(-self.noise, self.noise, (pc.shape[0], 1)) * normal
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
# parser.add_argument('--proj_dir', type=str, default="/mnt/disk6/wurundi/cad_gen",
#                    help="path to project folder where models and logs will be saved")
parser.add_argument('--proj_dir', type=str, default="/home/rundi/project_log/cad_gen",
                   help="path to project folder where models and logs will be saved")
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
        # for g in agent.optimizer.param_groups:
        #     g['lr'] = 1e-5

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

    # save_dir = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}_pc".format(args.ckpt, args.n_samples))
    save_dir = os.path.join(cfg.exp_dir, "results/pc2cad_ckpt{}_num{}".format(args.ckpt, args.n_samples))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_zs = []
    all_ids = []
    pbar = tqdm(test_loader)
    cnt = 0
    for i, data in enumerate(pbar):
        with torch.no_grad():
            pred_z, _ = agent.forward(data)
            pred_z = pred_z.detach().cpu().numpy()
            # print(pred_z.shape)
            all_zs.append(pred_z)

        all_ids.extend(data['id'])
        pts = data['points'].detach().cpu().numpy()
        # for j in range(pred_z.shape[0]):
        #     save_path = os.path.join(save_dir, "{}.ply".format(data['id'][j]))
        #     write_ply(pts[j], save_path)
        # for j in range(pred_z.shape[0]):
        #     save_path = os.path.join(save_dir, "{}.h5".format(data['id'][j]))
        #     with h5py.File(save_path, 'w') as fp:
        #         fp.create_dataset("zs", data=pred_z[j])

        cnt += pred_z.shape[0]
        if cnt > args.n_samples:
            break

    all_zs = np.concatenate(all_zs, axis=0)
    # save generated z
    save_path = os.path.join(cfg.exp_dir, "results/pc2cad_z_ckpt{}_num{}.h5".format(args.ckpt, args.n_samples))
    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("zs", shape=all_zs.shape, data=all_zs)

    save_path = os.path.join(cfg.exp_dir, "results/pc2cad_z_ckpt{}_num{}_ids.json".format(args.ckpt, args.n_samples))
    with open(save_path, 'w') as fp:
        json.dump(all_ids, fp)
