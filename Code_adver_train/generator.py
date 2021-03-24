from torch import nn
import torch

# Generator Code for noise generation
class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 32, 5, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.PReLU(),
            # state size. (ngf*32) x 5 x 5
            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.PReLU(),
            # state size. (ngf*16) x 10 x 19
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.PReLU(),
            # state size. (ngf*8) x 20 x 20
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.PReLU(),
            # state size. (ngf*4) x 40 x 40
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.PReLU(),
            # state size. (ngf*2) x 80 x 80
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.PReLU(),
            # state size. (ngf) x 160 x 160
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320 x 320
        )

    def forward(self, input):
        return self.main(input)
