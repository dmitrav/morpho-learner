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
from src import constants


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def run_autoencoder_training(data_loader_train, data_loader_test, model, optimizer, criterion, device, lr_scheduler=None, epochs=10):

    loss = 0
    val_loss = 0
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        for batch_features in data_loader_train:
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
        loss = loss / len(data_loader_train)

        val_loss = 0
        for batch_features in data_loader_test:
            batch_features = batch_features[0].float().to(device)
            outputs = model(batch_features)
            val_loss += criterion(outputs, batch_features).item()

        # compute the epoch training loss
        val_loss = val_loss / len(data_loader_test)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}, val_loss = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss, val_loss))

    return loss


def run_weakly_supervised_classifier_training(loader_train, loader_val, model, optimizer, criterion, device,
                                              lr_scheduler=None, epochs=10):
    f_acc = Accuracy().to(device)

    acc, val_acc = 0, -1
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        acc = 0
        for batch_features, batch_labels in loader_train:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()

            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_labels)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            # add the mini-batch training acc to epoch acc
            acc += float(f_acc(outputs, batch_labels))

        # compute epoch training loss
        loss = loss / len(loader_train)
        # compute epoch training accuracy
        acc = acc / len(loader_train)

        val_acc = 0
        for batch_features, batch_labels in loader_val:
            # process drugs data
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            val_acc += float(f_acc(outputs, batch_labels))

        # compute epoch training accuracy
        val_acc = val_acc / len(loader_val)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}, acc = {:.4f}, val_acc = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss, acc, val_acc))

    return acc


def run_supervised_classifier_training(loader_train, model, optimizer, criterion, device,
                                       lr_scheduler=None, epochs=10, test_loader=None):
    f_acc = Accuracy().to(device)
    print("training started...")
    train_acc, val_acc = 0, -1
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        train_acc = 0
        for batch_features, batch_labels in loader_train:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            with torch.enable_grad():
                train_loss = 0

                outputs = model(batch_features)
                train_loss += criterion(outputs, batch_labels.to(device))
                train_acc += float(f_acc(outputs, batch_labels.to(device)))

                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

        # compute epoch training loss
        loss = loss / len(loader_train)
        # compute epoch training accuracy
        train_acc = train_acc / len(loader_train)

        if test_loader is not None:
            val_acc = 0
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.float().to(device)
                outputs = model(batch_features)
                val_acc += float(f_acc(outputs, batch_labels.to(device)))
            # compute epoch training accuracy
            val_acc = val_acc / len(test_loader)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} sec, loss = {:.4f}, train_acc = {:.4f}, val_acc = {:.4f}"
              .format(epoch + 1, epochs, int(time.time() - start), loss, train_acc, val_acc))

    return train_acc, val_acc


def run_simultaneous_training(loader_train, loader_val, ae_model, ae_optimizer, ae_criterion, cl_model, cl_optimizer, cl_criterion,
                              device, epochs=10, ae_scheduler=None, cl_scheduler=None):

    f_acc = Accuracy().to(device)

    rec_loss_epoch = 0
    acc_epoch = 0
    for epoch in range(epochs):

        start = time.time()
        ae_loss_epoch = 0
        cl_loss_epoch = 0
        rec_loss_epoch = 0
        acc_epoch = 0
        for batch_features, batch_labels in loader_train:

            # TRAIN CLASSIFIER

            # reset gradients to zero
            cl_optimizer.zero_grad()
            # get features of drugs
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.to(device)
            # retrieve encodings
            encodings = ae_model.encoder(batch_features)
            # run through classifier
            outputs = cl_model(encodings)
            # calculate loss
            cl_loss = cl_criterion(outputs, batch_labels)
            cl_loss.backward()
            cl_optimizer.step()

            acc_epoch += float(f_acc(outputs, batch_labels))

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
                # add .1 of classifier loss
                ae_loss += 0.1 * cl_loss.item()

                # compute accumulated gradients
                ae_loss.backward()
                # perform parameter update based on current gradients
                ae_optimizer.step()

            # add the mini-batch training loss to epoch loss
            ae_loss_epoch += ae_loss.item()
            cl_loss_epoch += cl_loss.item()

        # compute the epoch training loss
        ae_loss_epoch = ae_loss_epoch / len(loader_train)
        cl_loss_epoch = cl_loss_epoch / len(loader_train)
        rec_loss_epoch = rec_loss_epoch / len(loader_train)
        acc_epoch = acc_epoch / len(loader_train)

        val_acc = 0
        for batch_features, batch_labels in loader_val:
            # process drugs
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.to(device)
            encodings = ae_model.encoder(batch_features)
            outputs = cl_model(encodings)
            val_acc += float(f_acc(outputs, batch_labels))

        # compute epoch validation accuracy
        val_acc = val_acc / len(loader_val)

        # update lr
        if ae_scheduler is not None:
            ae_scheduler.step()
        if cl_scheduler is not None:
            cl_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, ae_loss = {:.4f}, cl_loss = {:.4f}, rec_loss = {:.4f}, acc = {:.4f}, val_acc = {:.4f}"
              .format(epoch + 1, epochs, int((time.time() - start) / 60), ae_loss_epoch, cl_loss_epoch, rec_loss_epoch, acc_epoch, val_acc))

    return rec_loss_epoch, acc_epoch


def plot_reconstruction(data_loader, trained_model, save_to='res/', n_images=10):

    for i in range(n_images):
        train_features, _ = next(iter(data_loader))
        img = train_features[0].squeeze()
        img_tensor = torch.Tensor(numpy.expand_dims(train_features[0], axis=0))
        rec = trained_model(img_tensor.cuda())

        pyplot.figure()
        pyplot.subplot(121)
        pyplot.imshow(img, cmap="gray")
        pyplot.title("original")
        pyplot.subplot(122)
        pyplot.imshow(rec.cpu().detach().numpy()[0][0], cmap="gray")
        pyplot.title("reconstruction")

        if save_to is not None:
            if not os.path.exists(save_to + 'recs/'):
                os.makedirs(save_to + 'recs/')
            pyplot.savefig(save_to + 'recs/{}.pdf'.format(i))
        else:
            pyplot.show()
    pyplot.close('all')


def train_autoencoder(epochs, data_loader_train, data_loader_val, trained_ae=None, batch_size=256, device=torch.device('cuda')):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\ae\\'

    if trained_ae is not None:
        model = trained_ae
    else:
        model = Autoencoder().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()

    last_rec_loss = run_autoencoder_training(data_loader_train, data_loader_val, model, optimizer, criterion, device, epochs=epochs)

    save_path = save_path.replace('ae', 'ae_{}'.format(round(last_rec_loss, 4)))
    plot_reconstruction(data_loader_train, model, save_to=save_path, n_images=30)

    torch.save(model.state_dict(), save_path + 'autoencoder.torch')


def train_deep_classifier_weakly(epochs, loader_train, loader_val,
                                 trained_cl=None, device=torch.device('cuda')):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\dcl_weakly\\'

    if trained_cl is not None:
        model = trained_cl
    else:
        model = DeepClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    last_epoch_acc = run_weakly_supervised_classifier_training(loader_train, loader_val,
                                                               model, optimizer, criterion, device,
                                                               epochs=epochs)

    save_path = save_path.replace('dcl', 'dcl_{}'.format(round(last_epoch_acc, 4)))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + 'deep_classifier.torch')


def train_together(epochs, loader_train, loader_val,
                   trained_ae=None, trained_cl=None, device=torch.device('cuda')):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\aecl\\'

    if trained_ae is not None:
        ae = trained_ae
    else:
        ae = Autoencoder().to(device)

    ae_optimizer = optim.Adam(ae.parameters(), lr=0.0001)
    ae_criterion = nn.L1Loss()

    if trained_cl is not None:
        cl = trained_cl
    else:
        cl = Classifier().to(device)

    cl_optimizer = optim.Adam(cl.parameters(), lr=0.0001)
    cl_criterion = nn.CrossEntropyLoss()

    last_rec_loss, last_acc = run_simultaneous_training(loader_train, loader_val,
                                                        ae, ae_optimizer, ae_criterion,
                                                        cl, cl_optimizer, cl_criterion,
                                                        device, epochs=epochs)

    save_path = save_path.replace('aecl', 'aecl_{}_{}'.format(round(last_rec_loss, 4), round(last_acc, 4)))

    plot_reconstruction(loader_train, ae, save_to=save_path, n_images=30)
    torch.save(ae.state_dict(), save_path + 'ae.torch')
    torch.save(cl.state_dict(), save_path + 'cl.torch')


if __name__ == "__main__":

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'

    # define augmentations like in BYOL or SimCLR (almost)
    transform = torch.nn.Sequential(
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (.1, 2.0)), p=0.2),
        T.RandomResizedCrop((64, 64)),
        T.Normalize(mean=torch.tensor([0.449]), std=torch.tensor([0.226]))
    )

    # make a balanced dataset of drugs and controls
    data = MultiLabelDataset({0: path_to_controls, 1: path_to_drugs}, N=370000, transform=transform)
    training_data, validation_data = torch.utils.data.random_split(data, [700000, 40000])

    data_loader_train = DataLoader(training_data, batch_size=256, shuffle=True, num_workers=4)
    data_loader_val = DataLoader(validation_data, batch_size=256, shuffle=True, num_workers=4)

    print('training data:', training_data.__len__())
    print('validation data:', validation_data.__len__())

    train_ae_alone = True  # convolutional autoencoder
    train_cl_weakly = True  # weakly supervised classifier
    train_both_weakly = True  # autoencoder + weakly supervised classifier (2 classes)
    train_cl_with_byol = True  # train the common backbone with self-supervision as in BYOL

    device = torch.device('cuda')
    epochs = 100

    if train_ae_alone:
        tracemalloc.start()
        train_autoencoder(epochs, data_loader_train, data_loader_val, device=device)
        current, peak = tracemalloc.get_traced_memory()
        print('current: {} MB, peak: {} MB'.format(current / 10 ** 6, peak / 10 ** 6))
        tracemalloc.stop()

    if train_cl_weakly:
        tracemalloc.start()
        train_deep_classifier_weakly(epochs, data_loader_train, data_loader_val, device=device)
        current, peak = tracemalloc.get_traced_memory()
        print('current: {} MB, peak: {} MB'.format(current / 10 ** 6, peak / 10 ** 6))
        tracemalloc.stop()

    if train_both_weakly:
        tracemalloc.start()
        train_together(epochs, data_loader_train, data_loader_val, device=device)
        current, peak = tracemalloc.get_traced_memory()
        print('current: {} MB, peak: {} MB'.format(current / 10 ** 6, peak / 10 ** 6))
        tracemalloc.stop()

    if train_cl_with_byol:
        # change dataset's transform to identity, as BYOL already has it
        training_data.dataset.transform = torch.nn.Sequential(torch.nn.Identity())
        data_loader_train = DataLoader(training_data, batch_size=256, shuffle=True, num_workers=4)

        tracemalloc.start()
        run_training_for_64x64_cuts(epochs, data_loader_train, device=device)
        current, peak = tracemalloc.get_traced_memory()
        print('current: {} MB, peak: {} MB'.format(current / 10 ** 6, peak / 10 ** 6))
        tracemalloc.stop()
