import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.models import Autoencoder, Classifier
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier_with_pretrained_encoder
from src.trainer import train_deep_classifier_alone
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering

if __name__ == "__main__":
    train_deep_classifier_alone(60)