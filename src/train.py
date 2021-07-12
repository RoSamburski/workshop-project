import os.path
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
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
max_animation_length = 8

# number of color channels in images. We use colored images so 3
# We will actually do color reduction later as every image will use its' own palette.
nc = 3

# Size of input to generator.
# Since we wish to generate outputs based on existing inputs ("animate them"), and not generate new animations
# out of nothing, we will use meaningful inputs as opposed to randomized inputs.
# Input consists of:
# 1. A sample image ("Reference"),
# 2. Desired direction (Left/right) of result
# 3. Desired animation type
# 4. Desired number of animation frames
nz = (image_size**2)*nc + 3

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

# Dropout rate for multi-class Discriminator
dropout = 0.2

# Number of GPUs available. The computer used to training has one.
ngpu = 1


class FrameType(Enum):
    """ Enums for value representation """
    REF = 0
    IDLE = 1
    WALK = 2
    JUMP = 3


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


# Generator class, straight from example.
# modified feature spaces to also account for animation (additional image dimension)
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8 * max_animation_length, (4,), (1,), (0,), bias=False),
            nn.BatchNorm2d(ngf * 8 * max_animation_length),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4 x (mal)
            nn.ConvTranspose2d(ngf * 8 * max_animation_length, ngf * 4 * max_animation_length, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ngf * 4 * max_animation_length),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8 x (mal)
            nn.ConvTranspose2d(ngf * 4 * max_animation_length, ngf * 2 * max_animation_length, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ngf * 2 * max_animation_length),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16 x (mal)
            nn.ConvTranspose2d(ngf * 2, ngf, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ngf * max_animation_length),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32 x (mal)
            nn.ConvTranspose2d(ngf * max_animation_length, nc*max_animation_length, (4,), (2,), (1,), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64 x (mal)
        )

    def forward(self, input):
        return self.main(input)


# Discriminator class, straight from example
# Might need to add dropout after each LeakyReLu for multi-class
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 x (mal + 1 (reference)) + 2 (params)
            nn.Conv2d(nc * (max_animation_length+1) + 2, ndf, (4,), (2,), (1,), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, (4,), (1,), (0,), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class MulticlassDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(MulticlassDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 x (mal+1)
            nn.Conv2d(nc * (max_animation_length + 1), ndf, (4,), (2,), (1,), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Drop-out for classification
            nn.Dropout(p=dropout),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, (4,), (2,), (1,), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, (4,), (1,), (0,), bias=False),
            # TODO
            # We use semi-supervised learning, so we will utilise softmax
            nn.Softmax(),
        )

    def forward(self, input):
        return self.main(input)


class AnimationDataset(Dataset):
    """
    Dataset class made for loading our dataset.
    Every "data-point" is a folder containing:
    1. Reference image (titled REF.png)
    2. Animation frames (titled as numbers by order: 0.png to <MAL-1>.png)
    The labeling file contains the folder names, followed by the metadata of the animation (direction+type)
    """
    def __init__(self, labeling_file, root_dir, transform=None, target_transform=None):
        self.label_set = labeling_file  # TODO: read labeling file
        self.root_dir = root_dir
        self.data_set = self.build_dataset()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        char_path = Path(os.path.join(self.root_dir, self.data_set[idx]))
        item = np.ndarray((max_animation_length+1,))
        label = None  # TODO
        for image in char_path.iterdir():
            if "REF" in image.name.upper():
                # Reference image
                item[0] = read_image(os.path.join(self.root_dir, self.data_set[idx], image))
            else:
                index = int(image.name.split(".")[0])
                item[index+1] = read_image(os.path.join(self.root_dir, self.data_set[idx], image))
        return item, label

    def build_dataset(self):
        # TODO: put labeled examples first
        data_set = []
        for folder in Path(self.root_dir).iterdir():
            data_set.append(folder.name)
        return data_set


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
    ret = []
    assert root.is_dir(), "argument must specify a folder"
    for char in root.iterdir():
        if char.is_dir():
            reference = None
            animation = np.ndarray((max_animation_length,))
            for image in char.iterdir():
                if "REF" in image.name.upper():
                    # Reference image
                    reference = read_image(os.path.join(root, char, image))
                else:
                    index = int(image.name.split(".")[0])
                    animation[index] = read_image(os.path.join(root, char, image))
            ret.append((reference, animation))
    return np.ndarray(ret)


def main():
    # TODO: training
    pass


if __name__ == "__main__":
    main()
