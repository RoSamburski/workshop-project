import os.path
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
import torchvision.utils as vutils
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

import database_handler

# This implementation of DCGAN is based on the code in:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and modified to fit our needs.

""" Constants: """

dataset_root = database_handler.DATASET_ROOT

generator_inputs = database_handler.GENERATOR_INPUT

labels = database_handler.LABELS

model_folder = "models"

# Size of images
image_size = database_handler.IMAGE_SIZE

# maximum number of animation frames. We may generate less but never more.
max_animation_length = database_handler.MAX_ANIMATION_LENGTH

batch_size = 16

# number of color channels in images. We use RGBA images so 4
# We will actually do color reduction later as every image will use its' own palette.
nc = 4

# Size of input to noise-based generator.
# Since we wish to generate outputs based on existing inputs ("animate them"), and not generate new animations
# out of nothing, we want use meaningful inputs as opposed to randomized inputs.
# However, this doesn't give the generator nearly enough power.
# Input consists of:
# 1. A sample image ("Reference"),
# 2. Desired direction (Left/right) of result
# 3. Desired animation type
# 4. Desired number of animation frames
# Currently only supports 1, 3 and 4
nz = 100

# Size of label embedding dimension
embed_dim = 100

# Size of feature maps in generator
ngf = 32

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 500  # TODO: Try increasing the epoch num when running on Nadav's computer

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


# Generator class
# Layer-wise, it is similar to generator in DCGAN paper.
# However, it works on 4d tensors (because animation) and parameters were changed to fit.
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.label_transform = nn.Sequential(
            nn.Embedding(len(database_handler.AnimationType) * max_animation_length, embed_dim),
            nn.Linear(embed_dim, nc * (image_size ** 2))
        )
        self.model = nn.Sequential(
            # input is reference + encoded label, going into a convolution
            # kernel, stride, padding were chosen to not change 2nd to 4th dimension
            nn.ConvTranspose3d(3, ngf * 8,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (8*ngf, nc, image_size, image_size)
            nn.ConvTranspose3d(ngf * 8, ngf * 4,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state size. (4*ngf, nc, image_size, image_size)
            nn.ConvTranspose3d(ngf * 4, ngf * 2,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (2*ngf, nc, image_size, image_size)
            nn.ConvTranspose3d(ngf * 2, ngf,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf, nc, image_size, image_size)
            nn.ConvTranspose3d(ngf, max_animation_length,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False),
            nn.Tanh()
            # state size. (mal, nc, image_size, image_size)
        )

    def forward(self, input):
        image, label = input
        label = self.label_transform(label).view(-1, 2, nc, image_size, image_size)
        labeled_input = torch.cat((image, label), dim=1)
        out = self.model(labeled_input)
        # We want to force the generator to match the image to the given reference image
        return torch.cat((image, out), dim=1)


# A generator that makes entire animations from noise
class NoiseGenerator(nn.Module):
    def __init__(self, ngpu):
        super(NoiseGenerator, self).__init__()
        self.ngpu = ngpu
        self.label_transform = nn.Sequential(
            nn.Embedding(len(database_handler.AnimationType) * max_animation_length, embed_dim),
            nn.Linear(embed_dim, nz)
        )
        self.model = nn.Sequential(
            # input is noise + encoded label, going into a convolution
            nn.ConvTranspose3d(3 * nz, ngf * 8,
                               kernel_size=(2, 5, 5),
                               stride=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (8*ngf, 2, 5, 5)
            nn.ConvTranspose3d(ngf * 8, ngf * 4,
                               kernel_size=(2, 4, 4),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state size. (4*ngf, 2, 10, 10)
            nn.ConvTranspose3d(ngf * 4, ngf * 2,
                               kernel_size=(2, 4, 4),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (2*ngf, 2, 20, 20)
            nn.ConvTranspose3d(ngf * 2, ngf,
                               kernel_size=(2, 4, 4),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf, 2, 40, 40)
            nn.ConvTranspose3d(ngf, max_animation_length + 1,
                               kernel_size=(4, 4, 4),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               bias=False),
            nn.Tanh()
            # state size. (mal+1, 4, 80, 80)
        )

    def forward(self, input):
        noise, label = input
        label = self.label_transform(label).view(-1, 2 * nz, 1, 1, 1)
        labeled_input = torch.cat((noise, label), dim=1)
        out = self.model(labeled_input)
        # We want to force the generator to match the image to the given reference image
        return out


# Discriminator class, for single-class discrimination (real/fake)
# Layer-wise, it is similar to generator in DCGAN paper.
# However, it works on 4d tensors (because animation) and parameters were changed to fit.
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.label_transform = nn.Sequential(
            nn.Embedding(len(database_handler.AnimationType) * max_animation_length, embed_dim),
            nn.Linear(embed_dim, (max_animation_length + 1) * nc * (image_size ** 2))
        )
        self.model = nn.Sequential(
            # input dimensions are (2*(mal+1), nc, image_size, image_size)
            # The first dimension is "doubled" because of the label
            # Following are calculations for nc=4, image_size=80:
            nn.Conv3d(3 * (max_animation_length + 1), ndf,
                      kernel_size=(4, 4, 4),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1),
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf, 2, 40, 40)
            nn.Conv3d(ndf, ndf * 2,
                      kernel_size=(2, 4, 4),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (2*ndf, 2, 20, 20)
            nn.Conv3d(ndf * 2, ndf * 4,
                      kernel_size=(2, 4, 4),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (4*ndf, 2, 10, 10)
            nn.Conv3d(ndf * 4, ndf * 8,
                      kernel_size=(2, 4, 4),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (8*ndf, 2, 5, 5)
            nn.Conv3d(ndf * 8, 1,
                      kernel_size=(2, 5, 5),
                      stride=(1, 1, 1),
                      padding=(0, 0, 0),
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        image, label = input
        label = self.label_transform(label).view(-1, 2 * (max_animation_length + 1), nc, image_size, image_size)
        labeled_input = torch.cat((image, label), dim=1)
        return self.model(labeled_input)


# Might need to add dropout after each LeakyReLu for multi-class
class MulticlassDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(MulticlassDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input dimensions are (mal+1, nc, image_size, image_size)
            nn.Conv2d(nc * (max_animation_length + 1), ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Drop-out for classification
            nn.Dropout(p=dropout),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # TODO
            # We use semi-supervised learning, so we will utilise softmax
            nn.Softmax(),
        )

    def forward(self, input):
        return self.main(input)


def model_init(use_noise=False):
    if use_noise:
        G = NoiseGenerator(ngpu).to(device)
    else:
        G = Generator(ngpu).to(device)
    G.apply(weights_init)
    D = Discriminator(ngpu).to(device)
    D.apply(weights_init)
    return G, D


def generator_input_transform(reference: torch.Tensor, animation_type: database_handler.AnimationType, frame_count: int) \
        -> torch.Tensor:
    assert 1 <= frame_count <= max_animation_length
    return torch.cat([torch.flatten(reference),
                      torch.Tensor(animation_type.value),
                      torch.Tensor(frame_count)])


def dataset_transform(data):
    result_by_frame = [torchvision.transforms.Compose([torchvision.transforms.functional.to_tensor,
                                                       torchvision.transforms.CenterCrop(image_size)
                                                       ])(frame) for frame in data]
    result_tensor = torch.zeros((max_animation_length + 1, nc, image_size, image_size), dtype=result_by_frame[0].dtype)
    for i in range(len(result_by_frame)):
        result_tensor[i] = result_by_frame[i]
    return result_tensor


def single_class_training(use_noise=False):
    # INIT STEP:
    dataset = database_handler.AnimationDataset(root_dir=dataset_root, labeling_file=labels,
                                                transform=dataset_transform,
                                                target_transform=lambda x: torch.IntTensor([int(x[0].value),
                                                                                            int(x[1])]),
                                                use_palette_swap=True,
                                                )

    if use_noise:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        # To provide the generator with enough power,
        # we limit the discriminator to only learning from 75% of the database.
        discriminator_data = np.random.choice(len(dataset), size=3 * len(dataset) // 4, replace=False)
        dataloader = torch.utils.data.DataLoader(Subset(dataset, discriminator_data), batch_size=batch_size,
                                                 shuffle=True)

    # Create models and initialize loss function
    G, D = model_init(use_noise)
    loss_func = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # Label conventions
    real_label, fake_label = 1, 0
    # Things we keep track of to see progress
    img_examples = []
    # Make a fixed sample for visualization
    example_tags = [np.random.choice(database_handler.AnimationType),
                    np.random.randint(1, max_animation_length)]
    if use_noise:
        example_reference = torch.randn(1, nz, 1, 1, 1, device=device)
        print(example_tags)
    else:
        example_animation, _ = dataset[np.random.randint(0, len(dataset))]  # TODO
        plt.imshow(database_handler.IMAGE_TRANSFORM(example_animation[0]))
        example_reference = torch.zeros((1, 1, nc, image_size, image_size))
        example_reference[0, 0] = example_animation[0]
        example_reference = example_reference.to(device)
        plt.title("Animation type: {}\nFrames: {}".format(example_tags[0].name, example_tags[1]))
        plt.savefig("train_ref.png")
    example_tags = torch.IntTensor([example_tags[0].value, example_tags[1]]).to(device)

    G_losses = []
    D_losses = []
    iters = 0

    # TRAINING LOOP
    print("Begin Training on {}".format(device))
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data, data_labels) in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            D.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            real_cpu_labels = data_labels.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = D((real_cpu, real_cpu_labels)).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss_func(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch #
            # Take a random sample of reference images
            # generator_input = [generator_input_transform(dataset[np.random.randint(0, len(dataset))][0],
            #                                              np.random.choice(list(database_handler.AnimationType)),
            #                                              np.random.randint(1, max_animation_length),
            #                                              )
            #                    for i in range(b_size)]
            if use_noise:
                generator_input = torch.randn(b_size, nz, 1, 1, 1, device=device)
            else:
                generator_input = [torch.unsqueeze(dataset.get_reference_frame(np.random.randint(0, len(dataset))), 0)
                                   for _ in range(b_size)]
                generator_input = torch.stack(generator_input).to(device)
            generator_input_labels = torch.IntTensor([
                [np.random.choice(list(database_handler.AnimationType)).value,
                 np.random.randint(1, max_animation_length)]
                for _ in range(b_size)
            ]).to(device)
            # Generate fake image batch with G
            fake = G((generator_input, generator_input_labels))
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = D((fake.detach(), generator_input_labels)).view(-1)
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
            output = D((fake.detach(), generator_input_labels)).view(-1)
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
                    fake = G((example_reference, example_tags)).detach().cpu()[0]
                img_examples.append([database_handler.IMAGE_TRANSFORM(fake_frame) for fake_frame in fake])

            iters += 1

    # Plot result
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training-graph.png")
    # for i in range(len(img_examples)):
    #     for j in range(max_animation_length + 1):
    #         plt.subplot(4, 5, j + 1)
    #         plt.axis("off")
    #         plt.imshow(img_examples[i][j])
    #     plt.show()
    # Show final result
    for j in range(max_animation_length + 1):
        plt.subplot(4, 5, j + 1)
        plt.axis("off")
        plt.imshow(img_examples[-1][j])
    plt.savefig("final-output.png")
    name = "model-noise-out"
    torch.save({"generator:": G.state_dict(), "discriminator": D.state_dict()},
               os.path.join(model_folder, name + ".pt"))


def main():
    matplotlib.use("Agg")
    single_class_training(use_noise=True)


if __name__ == "__main__":
    main()
