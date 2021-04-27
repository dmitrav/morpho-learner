import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.constants import user
from src.models import Autoencoder, Classifier
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering

if __name__ == "__main__":
    # train_autoencoder()
    # train_classifier(10, batch_size=256, deep=True)

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    # path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\'
    path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\ae_0.6673\\'

    device = torch.device('cuda')

    # load trained autoencoder to use it in the transform
    ae = Autoencoder().to(device)
    # ae.load_state_dict(torch.load(path_to_ae_model + 'ae.torch', map_location=device))
    ae.load_state_dict(torch.load(path_to_ae_model + 'autoencoder.torch', map_location=device))
    ae.eval()

    transform = lambda x: ae.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

    # save_path = path_to_ae_model + 'drugs_clustering_mcs=10_ms=1\\'
    # plot_drugs_clustering(path_to_drugs, save_path, transform, min_cluster_size=10, min_samples=1)

    save_path = path_to_ae_model + 'cell_lines_clustering_mcs=10_ms=1\\'
    plot_cell_lines_clustering(path_to_drugs, path_to_controls, save_path, transform, min_cluster_size=10, min_samples=1)
