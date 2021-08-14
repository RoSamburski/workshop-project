import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

import database_handler
import train


batch_size = 128

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
            nn.Linear((database_handler.IMAGE_SIZE**2)*4, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, (database_handler.IMAGE_SIZE ** 2) * 4),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder():
    dataset = database_handler.AnimationDataset(
        labeling_file=database_handler.LABELS,
        root_dir=database_handler.DATASET_ROOT,
        transform=train.dataset_transform,
        use_palette_swap=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    autoencoder = Autoencoder(ngpu).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    for epoch in range(epochs):
        for data in dataloader:
            references = data[0][:, 0]
            print(references.shape)
            references = references.view(references.size(0), -1)
            references = Variable(references)
            output = autoencoder(references)

            loss = loss_func(output, references)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch [{}/{}], loss:{:.4f}".format(epoch+1, epochs, loss.data))

    sample = dataset.get_reference_frame(np.random.randint(0, len(dataset)))
    encoded_sample = autoencoder(Variable(sample.view(sample.size(0), -1)))
    encoded_sample = torch.reshape(encoded_sample, (4, database_handler.IMAGE_SIZE, database_handler.IMAGE_SIZE))
    plt.title("Before/after")
    plt.subplot(1, 2, 1)
    plt.imshow(sample)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(database_handler.IMAGE_TRANSFORM(encoded_sample))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    train_autoencoder()
