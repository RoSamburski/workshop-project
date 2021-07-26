import os.path
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.utils as vutils
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import database_handler

# This implementation of DCGAN is based on the code in:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and modified to fit our needs.

""" Constants: """

dataset_root = database_handler.DATASET_ROOT

generator_inputs = database_handler.GENERATOR_INPUT

labels = database_handler.LABELS

# Size of images
image_size = database_handler.IMAGE_SIZE

# maximum number of animation frames. We may generate less but never more.
max_animation_length = database_handler.MAX_ANIMATION_LENGTH

batch_size = 64

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

# Device training is performed on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


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


def model_init():
    G = Generator(ngpu).to(device)
    G.apply(weights_init)
    D = Discriminator(ngpu).to(device)
    D.apply(weights_init)
    return G, D


def dataloader_init():
    dataset = database_handler.AnimationDataset(root_dir=dataset_root, labeling_file=labels,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.Pad(image_size),
                                                    torchvision.transforms.CenterCrop(image_size)
                                                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def single_class_training():
    # INIT STEP:
    dataloader = dataloader_init()
    # Create models and initialize loss function
    G, D = model_init()
    loss_func = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # Label conventions
    real_label, fake_label = 1, 0

    # Things we keep track of to see progress
    img_examples = []
    example_reference = None  # TODO
    G_losses = []
    D_losses = []
    iters = 0

    # TRAINING LOOP
    print("Begin Training")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            D.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = D(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss_func(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch #
            # Generate batch of latent vectors
            # TODO: Set up random input
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = G(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = D(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss_func(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = D(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss_func(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = G(example_reference).detach().cpu()
                img_examples.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Plot result
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():
    single_class_training()


if __name__ == "__main__":
    main()
