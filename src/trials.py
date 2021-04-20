import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.constants import user
from src.models import Autoencoder, Classifier
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier

if __name__ == "__main__":
    # train_autoencoder()
    train_classifier(10, batch_size=256, deep=True)