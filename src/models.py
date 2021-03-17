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
            # nn.MaxPool2d(2),

            # nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 16, (3,3), stride=(1,1), padding=(1,1)),
            # nn.ReLU(True),
            # nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(16, 32, (3,3), stride=(1,1), padding=(1,1)),
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
        print('number of parameters: {}'.format(self.count_parameters()))

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)