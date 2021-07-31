import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm
from .base import BaseTrainer
from model.latentGAN import Discriminator, Generator
from utils import cycle


class TrainerLatentWGAN(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerLatentWGAN, self).__init__(cfg)
        self.batch_size = cfg.batch_size
        self.n_iters = cfg.n_iters
        self.critic_iters = cfg.critic_iters
        self.save_frequency = cfg.save_frequency
        self.gp_lambda = cfg.gp_lambda
        self.n_dim = cfg.n_dim

        # build netD and netG
        self.build_net(cfg)

        # set optimizer
        self.set_optimizer(cfg)

    def build_net(self, cfg):
        self.netD = Discriminator(cfg.h_dim, cfg.z_dim).cuda()
        self.netG = Generator(cfg.n_dim, cfg.h_dim, cfg.z_dim).cuda()

    def eval(self):
        self.netD.eval()
        self.netG.eval()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.9))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.9))

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.step))
            print("Saving checkpoint epoch {}...".format(self.clock.step))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'netG_state_dict': self.netG.cpu().state_dict(),
            'netD_state_dict': self.netD.cpu().state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
        }, save_path)

        self.netG.cuda()
        self.netD.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() # if use_cuda else alpha

        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda # LAMBDA
        return gradient_penalty

    def train(self, dataloader):
        """training process"""
        data = cycle(dataloader)

        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.cuda()
        mone = mone.cuda()

        pbar = tqdm(range(self.clock.step, self.n_iters))
        for iteration in pbar:
            ############################
            # (1) Update D network
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(self.critic_iters):
                real_data = next(data)

                real_data = real_data.cuda()
                real_data.requires_grad_(True)

                self.netD.zero_grad()

                # train with real
                D_real = self.netD(real_data)
                D_real = D_real.mean(dim=0, keepdim=True)
                D_real.backward(mone)

                # train with fake
                noise = torch.randn(self.batch_size, self.n_dim)
                noise = noise.cuda()
                fake = self.netG(noise).detach()
                inputv = fake
                D_fake = self.netD(inputv)
                D_fake = D_fake.mean(dim=0, keepdim=True)
                D_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = self.calc_gradient_penalty(self.netD, real_data, fake.data)
                gradient_penalty.backward()

                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                self.optimizerD.step()

            # if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation
            self.netG.zero_grad()

            noise = torch.randn(self.batch_size, self.n_dim)
            noise = noise.cuda()
            noise.requires_grad_(True)

            fake = self.netG(noise)
            G = self.netD(fake)
            G = G.mean(dim=0, keepdim=True)
            G.backward(mone)
            G_cost = -G
            self.optimizerG.step()

            # Write logs and save samples
            pbar.set_postfix({"D_loss": D_cost.item(), "G_loss": G_cost.item()})
            self.train_tb.add_scalars("loss", {"D_loss": D_cost.item(), "G_loss": G_cost.item()}, global_step=self.clock.step)
            self.train_tb.add_scalar("wasserstein distance", Wasserstein_D.item(), global_step=self.clock.step)

            # save model
            self.clock.tick()
            if self.clock.step % self.save_frequency == 0:
                self.save_ckpt()

    def generate(self, n_samples, return_score=False):
        """generate samples"""
        self.eval()

        chunk_num = n_samples // self.batch_size
        generated_z = []
        z_scores = []
        for i in range(chunk_num):
            noise = torch.randn(self.batch_size, self.n_dim).cuda()
            with torch.no_grad():
                fake = self.netG(noise)
                G_score = self.netD(fake)
            G_score = G_score.detach().cpu().numpy()
            fake = fake.detach().cpu().numpy()
            generated_z.append(fake)
            z_scores.append(G_score)
            print("chunk {} finished.".format(i))

        remains = n_samples - self.batch_size * chunk_num
        noise = torch.randn(remains, self.n_dim).cuda()
        with torch.no_grad():
            fake = self.netG(noise)
            G_score = self.netD(fake)
            G_score = G_score.detach().cpu().numpy()
            fake = fake.detach().cpu().numpy()
        generated_z.append(fake)
        z_scores.append(G_score)

        generated_z = np.concatenate(generated_z, axis=0)
        z_scores = np.concatenate(z_scores, axis=0)
        if return_score:
            return generated_z, z_scores
        else:
            return generated_z