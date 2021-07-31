import os
from utils import ensure_dirs
import argparse
import json
import shutil


class ConfigLGAN(object):
    def __init__(self):
        self.set_configuration()

        # parse command line arguments
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.data_root = os.path.join(args.proj_dir, args.exp_name, "results/all_zs_ckpt{}.h5".format(args.ae_ckpt))
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "lgan_{}".format(args.ae_ckpt))
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')

        if (not args.test) and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if not args.test:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(self.__dict__, f, indent=2)

    def set_configuration(self):
        # network configuration
        self.n_dim = 64
        self.h_dim = 512
        self.z_dim = 256

        # WGAN-gp configuration
        self.beta1 = 0.5
        self.critic_iters = 5
        self.gp_lambda = 10

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--proj_dir', type=str, default="proj_log",
                            help="path to project folder where models and logs will be saved")
        parser.add_argument('--exp_name', type=str, required=True, help="name of this experiment")
        parser.add_argument('--ae_ckpt', type=str, required=True, help="ckpt for autoencoder")
        parser.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--test', action='store_true', help="test mode")
        parser.add_argument('--n_samples', type=int, default=100, help="number of samples to generate when testing")
        parser.add_argument('-g', '--gpu_ids', type=str, default="0",
                            help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

        parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--n_iters', type=int, default=200000, help="total number of iterations to train")
        parser.add_argument('--save_frequency', type=int, default=100000, help="save models every x iterations")
        parser.add_argument('--lr', type=float, default=2e-4, help="initial learning rate")

        args = parser.parse_args()
        return parser, args
