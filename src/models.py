import torch, numpy, pandas
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.constants import cell_lines, drugs


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),  # 128 x 4 x 4
        )

        self.decoder = nn.Sequential(

            nn.Upsample(scale_factor=4),
            nn.ConvTranspose2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=(1,1), padding=(1,1))
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),  # 128 x 4 x 4

            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        # plugs in after convolutional autoencoder -> needs flattening of the filters
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
