import torch, numpy, pandas
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler


class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 32, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 8, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 32, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(32, 128, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)