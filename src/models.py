import torch, numpy, pandas
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler


class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.Conv2d(16, 8, (3,3), stride=(1,1), padding=(1,1))
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(8, 32, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Classifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
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


class DeepClassifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2048, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 2),
            nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

