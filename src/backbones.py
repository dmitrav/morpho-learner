
import os, pandas, time, torch, numpy, itertools, random, tracemalloc
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchmetrics import Accuracy
import torch.nn.functional as F
from torchvision import transforms as T

from src.models import Autoencoder, Classifier, DeepClassifier
from src.byol import run_training_for_64x64_cuts
from src.datasets import CustomImageDataset, JointImageDataset, MultiLabelDataset
from src import constants, trainer


# TODO:
#  - conv layer in the end?
#  - av / min pooling?
#  - deeper?


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(1, 1)),  # 8 x 16 x 16

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


class Backbone_2a(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 32 x 8 x 8

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


class Backbone_2c(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),

            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # 32 x 8 x 8

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


class Backbone_2b(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),   # 32 x 8 x 8

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


class Backbone_2d(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),

            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # 32 x 8 x 8

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


class Backbone_3a(nn.Module):

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
            nn.MaxPool2d(4),   # 128 x 4 x 4

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


class Backbone_3e(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.AvgPool2d(4),   # 128 x 4 x 4

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


class Backbone_3c(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),

            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),

            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),  # 128 x 4 x 4

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


class Backbone_3b(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
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


class Backbone_3d(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),

            nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4),

            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1)),  # 128 x 4 x 4

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


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


if __name__ == '__main__':

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'

    # define augmentations like in BYOL or SimCLR (almost)
    transform = torch.nn.Sequential(
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (.1, 2.0)), p=0.2),
        T.RandomResizedCrop((64, 64)),
        T.Normalize(mean=torch.tensor([0.449]), std=torch.tensor([0.226]))
    )
    # define no transform
    no_transform = torch.nn.Sequential(torch.nn.Identity())

    # make a balanced dataset of drugs and controls
    data = MultiLabelDataset({0: path_to_controls, 1: path_to_drugs}, shuffle=True)
    training_data, validation_data = torch.utils.data.random_split(data, [700000, 99577])

    device = torch.device('cuda')
    epochs = 20

    models = [
        Backbone_3e,
        Backbone_3a
    ]

    for T in [no_transform, transform]:

        training_data.dataset.transform = T
        validation_data.dataset.transform = T
        data_loader_train = DataLoader(training_data, batch_size=256, shuffle=False, num_workers=4)
        data_loader_val = DataLoader(validation_data, batch_size=256, shuffle=False, num_workers=4)

        print('training data:', training_data.__len__())
        print('validation data:', validation_data.__len__())

        for model in models:

            model = model().to(device)

            tracemalloc.start()
            trainer.train_deep_classifier_weakly(epochs, data_loader_train, data_loader_val, trained_cl=model, device=device)
            current, peak = tracemalloc.get_traced_memory()
            print('current: {} MB, peak: {} MB'.format(current / 10 ** 6, peak / 10 ** 6))
            tracemalloc.stop()
