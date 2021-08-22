import os.path

import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable

import database_handler
import gan_model

autoencoder_file = "conv-autoencoder.pt"

batch_size = 16

epochs = 100

learning_rate = 1e-3

weight_decay = 1e-5

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, ngpu=1):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu

        self.encoder = nn.Sequential(
            nn.Linear((database_handler.IMAGE_SIZE**2)*gan_model.nc, 8*gan_model.nz),
            nn.ReLU(True),
            nn.Linear(8*gan_model.nz, 4*gan_model.nz),
            nn.ReLU(True),
            nn.Linear(4*gan_model.nz, 2*gan_model.nz),
            nn.ReLU(True),
            nn.Linear(2*gan_model.nz, gan_model.nz),
        )

        self.decoder = nn.Sequential(
            nn.Linear(gan_model.nz, 2*gan_model.nz),
            nn.ReLU(True),
            nn.Linear(2*gan_model.nz, 4*gan_model.nz),
            nn.ReLU(True),
            nn.Linear(4*gan_model.nz, 8*gan_model.nz),
            nn.ReLU(True),
            nn.Linear(8*gan_model.nz, (database_handler.IMAGE_SIZE**2)*gan_model.nc),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, ngpu=1):
        super(ConvolutionalAutoencoder, self).__init__()
        self.ngpu = ngpu

        self.encoder = nn.Sequential(
            # Input: (4, 64, 64)
            nn.Conv2d(3, 4*gan_model.nz,
                      kernel_size=(4, 4),
                      stride=(4, 4),
                      padding=(1, 1)
                      ),
            nn.ReLU(True),
            # State: (400, 16, 16)
            nn.Conv2d(4*gan_model.nz, 2*gan_model.nz,
                      kernel_size=(4, 4),
                      stride=(4, 4),
                      padding=(1, 1)
                      ),
            nn.ReLU(True),
            # State: (200, 4, 4)
            nn.Conv2d(2*gan_model.nz, gan_model.nz,
                      kernel_size=(4, 4),
                      stride=(4, 4),
                      padding=(1, 1)
                      ),
            # Out: (100, 1, 1)
        )

        self.decoder = nn.Sequential(
            # Input: (100, 1, 1)
            nn.ConvTranspose2d(gan_model.nz, 2*gan_model.nz,
                               kernel_size=(4, 4),
                               stride=(4, 4),
                               padding=(0, 0),
                               ),
            nn.ReLU(True),
            # State: (200, 4, 4)
            nn.ConvTranspose2d(2*gan_model.nz, 4*gan_model.nz,
                               kernel_size=(4, 4),
                               stride=(4, 4),
                               padding=(0, 0),
                               ),
            nn.ReLU(True),
            # State: (400, 16, 16)
            nn.ConvTranspose2d(4*gan_model.nz, gan_model.nc,
                               kernel_size=(4, 4),
                               stride=(4, 4),
                               padding=(0, 0),
                               ),
            nn.Tanh(),
            # Out: (3, 64, 64)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder():
    dataset = database_handler.AnimationDataset(
        labeling_file=database_handler.LABELS,
        root_dir=database_handler.DATASET_ROOT,
        transform=gan_model.dataset_transform,
        target_transform=lambda x: torch.IntTensor([int(x[0].value),
                                                    int(x[1])]),
        use_palette_swap=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    autoencoder = ConvolutionalAutoencoder(ngpu).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    for epoch in range(epochs):
        for data in dataloader:
            references = data[0][:, 0].to(device)
            references = references.view(references.size(0), -1)
            references = Variable(references)
            output = autoencoder(references)

            loss = loss_func(output, references)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch [{}/{}], loss:{:.4f}".format(epoch+1, epochs, loss.data))

    sample, _ = dataset[np.random.randint(0, len(dataset))]
    sample = sample[0].to(device)
    encoded_sample = sample.view(-1)
    encoded_sample = Variable(encoded_sample)
    encoded_sample = autoencoder(encoded_sample)
    encoded_sample = torch.reshape(encoded_sample, (4, database_handler.IMAGE_SIZE, database_handler.IMAGE_SIZE))
    plt.title("Before/after")
    plt.subplot(1, 2, 1)
    plt.imshow(database_handler.IMAGE_TRANSFORM(sample))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(database_handler.IMAGE_TRANSFORM(encoded_sample))
    plt.axis("off")
    plt.savefig("conv-autoencoder-difference.png")
    torch.save(autoencoder.state_dict(), os.path.join(gan_model.model_folder, autoencoder_file))


if __name__ == "__main__":
    matplotlib.use("Agg")
    train_autoencoder()
