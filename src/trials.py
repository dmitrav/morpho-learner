import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.models import Autoencoder, Classifier
from src.trainer import CustomImageDataset, JointImageDataset
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier_with_pretrained_encoder
from src.trainer import train_deep_classifier_alone
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering

if __name__ == "__main__":

    import torch
    from vit_pytorch import ViT, Dino

    device = torch.device('cuda')

    model = ViT(
        image_size=64,
        patch_size=8,
        num_classes=2,
        dim=256,
        depth=3,
        heads=4,
        mlp_dim=512
    ).to(device)

    learner = Dino(
        model,
        image_size=64,
        hidden_layer='to_latent',  # hidden layer name or index, from which to extract the embedding
        projection_hidden_size=64,  # projector network hidden dimension
        projection_layers=4,  # number of layers in projection network
        num_classes_K=2048,  # output logits dimensions (referenced as K in paper)
        student_temp=0.9,  # student temperature
        teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
        global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay=0.9,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay=0.9,
        # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    ).to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=0.0001)

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\dino\\'

    batch_size = 512
    Nd, Nc = 50000, 50000  # ~8.9%

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)
    training_drugs, _ = torch.utils.data.random_split(training_drugs, [Nd, training_drugs.__len__() - Nd])

    training_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)
    training_controls, _ = torch.utils.data.random_split(training_controls, [Nc, training_controls.__len__() - Nc])

    joint_data = JointImageDataset([training_drugs, training_controls], transform=lambda x: x / 255.)

    data_loader = DataLoader(joint_data, batch_size=batch_size, shuffle=True)

    for epoch in range(100):
        start = time.time()
        epoch_loss = 0
        for batch_features in data_loader:
            images = batch_features[0].float().to(device)
            loss = learner(images)
            epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

        epoch_loss = epoch_loss / len(data_loader)
        print("epoch {}: {} min, loss = {:.4f}".format(epoch + 1, int((time.time() - start) / 60), epoch_loss))

    # # save your improved network
    # torch.save(model.state_dict(), './pretrained-net.pt')
