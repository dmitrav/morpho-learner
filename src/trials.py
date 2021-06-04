import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp
from torch.nn import Sequential
from vit_pytorch import ViT, Dino

from src.models import Autoencoder, Classifier, DeepClassifier
from src.trainer import CustomImageDataset, JointImageDataset
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier_with_pretrained_encoder
from src.trainer import train_deep_classifier_alone
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering
from src import dino


if __name__ == "__main__":

    model_path = 'D:\ETH\projects\morpho-learner\\res\dino\\f7f44e6e\\'
    data_path = 'D:\ETH\projects\morpho-learner\data\cut\\'
    dino.cluster_images(model_path, data_path)