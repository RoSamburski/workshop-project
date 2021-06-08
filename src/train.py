from typing import Tuple
from enum import Enum

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# This implementation of DCGAN is based on the code in:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and modified to fit our needs.

""" Constants: """

dataset_root = "data"

batch_size = 64

# Size of images
image_size = 64

# maximum number of animation frames. We may generate less but never more.
max_animation_length = 16

# number of color channels in images. We use colored images so 3
# We will actually do color reduction later as every image will use its' own palette.
nc = 3

# Size of input to generator.
# Since we wish to generate outputs based on existing inputs ("animate them"), and not generate new animations
# out of nothing, we will use meaningful inputs as opposed to randomized inputs.
# Input consists of:
# 1. A sample image ("Reference"),
# 2. Desired direction (Left/right) of result
# 3. Desired number of animation frames
nz = (image_size**2)*nc + 2

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. The computer used to training has one.
ngpu = 1


""" Enums for value representation """
class FrameType(Enum):
    REF = 0
    WALK = 1
    JUMP = 2


class Direction(Enum):
    L = 0
    R = 1


# Weight initialization, as recommended
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator class, straight from example
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator class, straight from example
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def load_data(root_path: str) -> np.array:
    """
    Loads the data from the database.
    Entries are folders of the following structure:
    [name]
    ->
    :param root_path: Path to the database's root.
    :return:
    """
    root = Path(root_path)
    assert root.is_dir(), "argument must specify a folder"
    for char in root.iterdir():
        if char.is_dir():
            pass
    return


def main():
    pass


if __name__ == "__main__":
    main()
