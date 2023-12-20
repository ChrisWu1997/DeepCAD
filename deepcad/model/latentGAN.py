import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self, n_dim, h_dim, z_dim):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(n_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, z_dim),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        output = torch.tanh(output)
        return output


class Discriminator(nn.Module):

    def __init__(self, h_dim, z_dim):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)
