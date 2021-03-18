import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.constants import user
from src.models import Autoencoder, Classifier
from src.trainer import plot_reconstruction


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = pandas.DataFrame({'img': [f for f in os.listdir(img_dir)],
                                            'label': [label for f in os.listdir(img_dir)]})

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample


def run_simultaneous_training(data_loader_drugs, data_loader_controls,
                              ae_model, ae_optimizer, ae_criterion,
                              cl_model, cl_optimizer, cl_criterion,
                              device, epochs=10):

    for epoch in range(epochs):

        start = time.time()
        ae_loss_epoch = 0
        cl_loss_epoch = 0
        rec_loss_epoch = 0
        acc_epoch = 0
        for batch_features, batch_labels in data_loader_drugs:

            # TRAIN CLASSIFIER

            # reset gradients to zero
            cl_optimizer.zero_grad()

            # get features of drugs
            batch_features = batch_features.float().to(device)
            # retrieve encodings
            encodings = ae_model.encoder(batch_features)
            # reshape fo classifier input
            encodings = torch.reshape(encodings, (encodings.shape[0], -1))
            # run through classifier
            outputs = cl_model(encodings)
            # calculate loss on drugs
            cl_loss = cl_criterion(outputs, batch_labels)
            true_negatives = (outputs.argmax(-1) == 0).float().detach().numpy()

            # get features of controls
            control_features, control_labels = next(iter(data_loader_controls))
            control_features = control_features.float().to(device)
            # retrieve encodings
            encodings = ae_model.encoder(control_features)
            # reshape fo classifier input
            encodings = torch.reshape(encodings, (encodings.shape[0], -1))
            # run through classifier
            outputs = cl_model(encodings)
            true_positives = (outputs.argmax(-1) == 1).float().detach().numpy()
            # add loss on controls
            cl_loss += cl_criterion(outputs, control_labels)

            cl_loss.backward()
            cl_optimizer.step()

            acc_epoch += (true_positives.sum() + true_negatives.sum()) / len(true_positives) / len(true_negatives)

            # TRAIN AUTOENCODER

            # reset the gradients to zero
            ae_optimizer.zero_grad()
            with torch.enable_grad():

                ae_loss = 0.
                # compute reconstructions
                outputs = ae_model(batch_features)
                # compute training reconstruction loss
                ae_loss += ae_criterion(outputs, batch_features)
                rec_loss_epoch += ae_loss.item()
                # add 1/2 of classifier loss
                ae_loss += 0.5 * cl_loss.item()

                # compute accumulated gradients
                ae_loss.backward()
                # perform parameter update based on current gradients
                ae_optimizer.step()

            # add the mini-batch training loss to epoch loss
            ae_loss_epoch += ae_loss.item()
            cl_loss_epoch += cl_loss.item()

        # compute the epoch training loss
        ae_loss_epoch = ae_loss_epoch / len(data_loader_drugs)
        cl_loss_epoch = cl_loss_epoch / len(data_loader_drugs)
        rec_loss_epoch = rec_loss_epoch / len(data_loader_drugs)
        acc_epoch = acc_epoch / len(data_loader_drugs)

        # display the epoch training loss
        print("epoch {}/{}: {} sec, ae_loss = {:.4f}, cl_loss = {:.4f}, rec_loss = {:.4f}, acc = {:.4f}"
              .format(epoch + 1, epochs, int(time.time() - start), ae_loss_epoch, cl_loss_epoch, rec_loss_epoch, acc_epoch))


if __name__ == "__main__":
    path_to_drugs = '/Users/{}/ETH/projects/morpho-learner/data/cut/'.format(user)
    path_to_controls = '/Users/{}/ETH/projects/morpho-learner/data/cut_controls/'.format(user)
    save_path = '/Users/{}/ETH/projects/morpho-learner/res/aecl/'.format(user)

    device = torch.device("cpu")

    ae = Autoencoder().to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.0003)
    ae_criterion = nn.BCELoss()

    cl = Classifier().to(device)
    cl_optimizer = optim.Adam(cl.parameters(), lr=0.0003)
    cl_criterion = nn.CrossEntropyLoss()

    training_data_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)
    training_data_drugs, test_data_drugs = torch.utils.data.random_split(training_data_drugs, [5000, 75000])

    training_data_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)

    data_loader_train_drugs = DataLoader(training_data_drugs, batch_size=64, shuffle=True)
    data_loader_train_controls = DataLoader(training_data_controls, batch_size=64, shuffle=True)

    run_simultaneous_training(data_loader_train_drugs, data_loader_train_controls,
                              ae, ae_optimizer, ae_criterion,
                              cl, cl_optimizer, cl_criterion,
                              device, 50)

    plot_reconstruction(data_loader_train_drugs, ae, save_to=save_path, n_images=50)
    torch.save(ae.state_dict(), save_path + 'aecl.torch')

