import os
import argparse
import json
import shutil

from deepcad.cadlib.macro import *
from deepcad.utils import ensure_dirs


class ConfigAE(object):
    def __init__(self, phase, **kwargs):
        self.is_train = phase == "train"

        self._set_configuration()
        self._set_default()

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and self.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if self.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)

        # create soft link to experiment log directory
        # if not os.path.exists('train_log'):
            # os.symlink(self.exp_dir, 'train_log')

        # save this configuration
        if self.is_train:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(self.__dict__, f, indent=2)

    def _set_configuration(self):
        self.args_dim = ARGS_DIM # 256
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)  # line, arc, circle, EOS, SOS

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality
        self.dropout = 0.1                # Dropout rate used in basic layers and Transformers
        self.dim_z = 256                 # Latent vector dimensionality
        self.use_group_emb = True

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES

        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }

    def _set_default(self):
        self.proj_dir = "proj_log"
        self.data_root = "data"
        self.exp_name = os.getcwd().split('/')[-1]
        self.gpu_ids = '0'
        self.batch_size = 512
        self.num_workers = 8
        self.nr_epochs = 1000
        self.lr = 1e-3
        self.grad_clip = 1.0
        self.warmup_step = 2000
        self.cont = False
        self.ckpt = 'latest'
        self.vis = False
        self.save_frequency = 500
        self.val_frequency = 10
        self.vis_frequency = 2000
        self.augment = False
        self.mode = None if self.is_train else 'rec'
        self.outputs = None
        self.z_path = None


    def print_config(self):
        print("----Experiment Configuration-----")
        for attr in self.__dict__:
            print("{0:20}".format(attr), getattr(self, attr))

    # TODO: Move this CLI to a dedicated script
    # def parse(self):
    #     """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
    #     parser = argparse.ArgumentParser()

    #     parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
    #     parser.add_argument('--data_root', type=str, default="data", help="path to source data folder")
    #     parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
    #     parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    #     parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    #     parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

    #     parser.add_argument('--nr_epochs', type=int, default=1000, help="total number of epochs to train")
    #     parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    #     parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
    #     parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
    #     parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
    #     parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    #     parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    #     parser.add_argument('--save_frequency', type=int, default=500, help="save models every x epochs")
    #     parser.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
    #     parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
    #     parser.add_argument('--augment', action='store_true', help="use random data augmentation")
        
    #     if not self.is_train:
    #         parser.add_argument('-m', '--mode', type=str, choices=['rec', 'enc', 'dec'])
    #         parser.add_argument('-o', '--outputs', type=str, default=None)
    #         parser.add_argument('--z_path', type=str, default=None)
        
    #     args = parser.parse_args()
    #     return parser, args
