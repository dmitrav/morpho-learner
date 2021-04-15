import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.constants import user
from src.models import Autoencoder, Classifier


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


def run_autoencoder_training(data_loader, model, optimizer, criterion, device, lr_scheduler=None, epochs=10):

    for epoch in range(epochs):

        start = time.time()
        loss = 0
        for batch_features in data_loader:
            # load it to the active device
            batch_features = batch_features[0].float().to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            # compute reconstructions
            outputs = model(batch_features)
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(data_loader)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss))


def run_classifier_training(data_loader_drugs, data_loader_controls, model, optimizer, criterion, device, lr_scheduler=None, epochs=10):

    for epoch in range(epochs):

        start = time.time()
        loss = 0
        acc = 0
        for batch_features, batch_labels in data_loader_drugs:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            with torch.enable_grad():
                train_loss = 0

                # process drugs data
                outputs = model(batch_features)
                train_loss += criterion(outputs, batch_labels)
                true_negatives = (outputs.argmax(-1) == 0).float().detach().numpy()

                # process controls data
                batch_features, batch_labels = next(iter(data_loader_controls))
                outputs = model(batch_features.float().to(device))
                train_loss += criterion(outputs, batch_labels)
                true_positives = (outputs.argmax(-1) == 1).float().detach().numpy()

                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

        # compute epoch training loss
        loss = loss / len(data_loader_drugs)
        # compute epoch accuracy
        acc = acc / len(data_loader_drugs)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} sec, loss = {:.4f}, acc = {:.4f}".format(epoch + 1, epochs, int(time.time() - start), loss, acc))


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
            cl_loss = cl_criterion(outputs, batch_labels.to(device))
            true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()

            # get features of controls
            control_features, control_labels = next(iter(data_loader_controls))
            control_features = control_features.float().to(device)
            # retrieve encodings
            encodings = ae_model.encoder(control_features)
            # reshape fo classifier input
            encodings = torch.reshape(encodings, (encodings.shape[0], -1))
            # run through classifier
            outputs = cl_model(encodings)
            true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()
            # add loss on controls
            cl_loss += cl_criterion(outputs, control_labels.to(device))

            cl_loss.backward()
            cl_optimizer.step()

            acc_epoch += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

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
        print("epoch {}/{}: {} min, ae_loss = {:.4f}, cl_loss = {:.4f}, rec_loss = {:.4f}, acc = {:.4f}"
              .format(epoch + 1, epochs, int((time.time() - start) / 60), ae_loss_epoch, cl_loss_epoch, rec_loss_epoch, acc_epoch))


def plot_reconstruction(data_loader, trained_model, save_to='res/', n_images=10):

    for i in range(n_images):
        train_features, _ = next(iter(data_loader))
        img = train_features[0].squeeze()
        img_tensor = torch.Tensor(numpy.expand_dims(train_features[0], axis=0))
        rec = trained_model(img_tensor)

        pyplot.figure()
        pyplot.subplot(121)
        pyplot.imshow(img, cmap="gray")
        pyplot.title("original")
        pyplot.subplot(122)
        pyplot.imshow(rec.detach().numpy()[0][0], cmap="gray")
        pyplot.title("reconstruction")

        if save_to is not None:
            if not os.path.exists(save_to + 'recs/'):
                os.makedirs(save_to + 'recs/')
            pyplot.savefig(save_to + 'recs/{}.pdf'.format(i))
        else:
            pyplot.show()


def train_autoencoder():

    path_to_data = 'D:\ETH\projects\morpho-learner\data\cut\\'
    save_path = 'D:\ETH\projects\morpho-learner\data\\res\\ae\\'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = 200000  # like 5 images of each drug

    training_data = CustomImageDataset(path_to_data, 0, transform=lambda x: x / 255.)
    training_data, test_data = torch.utils.data.random_split(training_data, [N, training_data.__len__() - N])

    device = torch.device("cpu")
    data_loader_train = DataLoader(training_data, batch_size=64, shuffle=True)
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.5)
    criterion = nn.BCELoss()

    # best loss = 0.6331
    run_autoencoder_training(data_loader_train, model, optimizer, criterion, device, lr_scheduler=scheduler, epochs=30)
    plot_reconstruction(data_loader_train, model, save_to=save_path, n_images=20)

    torch.save(model.state_dict(), save_path + 'autoencoder.torch')


def train_classifier():

    path_to_drugs = '/Users/{}/ETH/projects/morpho-learner/data/cut/'.format(user)
    path_to_controls = '/Users/{}/ETH/projects/morpho-learner/data/cut_controls/'.format(user)
    save_path = '/Users/{}/ETH/projects/morpho-learner/res/'.format(user)

    device = torch.device("cpu")

    # load trained autoencoder to use it in the transform
    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(save_path+'autoencoder.torch', map_location=device))
    ae.eval()

    transform = lambda x: ae.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0))).reshape(-1)
    training_data_drugs = CustomImageDataset(path_to_drugs, 0, transform=transform)
    training_data_drugs, test_data_drugs = torch.utils.data.random_split(training_data_drugs, [10000, 70000])

    training_data_controls = CustomImageDataset(path_to_controls, 1, transform=transform)

    data_loader_train_drugs = DataLoader(training_data_drugs, batch_size=64, shuffle=True)
    data_loader_train_controls = DataLoader(training_data_controls, batch_size=64, shuffle=True)

    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    run_classifier_training(data_loader_train_drugs, data_loader_train_controls,
                            model, optimizer, criterion, device, epochs=10)

    torch.save(model.state_dict(), save_path + 'classifier.torch')


def train_together():

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\data\\res\\aecl\\'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ae = Autoencoder().to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.0003)
    ae_criterion = nn.BCELoss()

    cl = Classifier().to(device)
    cl_optimizer = optim.Adam(cl.parameters(), lr=0.0006)
    cl_criterion = nn.CrossEntropyLoss()

    N = 200000  # like 5 images of each drug

    training_data_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)
    training_data_drugs, _ = torch.utils.data.random_split(training_data_drugs, [N, training_data_drugs.__len__() - N])

    training_data_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)
    training_data_controls, _ = torch.utils.data.random_split(training_data_controls, [N, training_data_controls.__len__() - N])

    print('total drugs:', training_data_drugs.__len__())
    print('total controls:', training_data_controls.__len__(), '\n')

    data_loader_train_drugs = DataLoader(training_data_drugs, batch_size=64, shuffle=True)
    data_loader_train_controls = DataLoader(training_data_controls, batch_size=64, shuffle=True)

    run_simultaneous_training(data_loader_train_drugs, data_loader_train_controls,
                              ae, ae_optimizer, ae_criterion,
                              cl, cl_optimizer, cl_criterion,
                              device, 30)

    plot_reconstruction(data_loader_train_drugs, ae, save_to=save_path, n_images=30)
    torch.save(ae.state_dict(), save_path + 'aecl1.torch')
    torch.save(cl.state_dict(), save_path + 'aecl2.torch')


if __name__ == "__main__":
    # train_autoencoder()
    # train_classifier()
    train_together()
    pass
