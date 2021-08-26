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
from src.datasets import CustomImageDataset, JointImageDataset, MultiLabelDataset, MultiCropDataset
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


def run_autoencoder_training(data_loader_train, data_loader_test, model, optimizer, criterion, device,
                             lr_scheduler=None, epochs=10, save_path="..\\res\\"):

    loss_history = []
    val_loss_history = []
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        n_crops = 1
        for batch in data_loader_train:

            n_crops = len(batch)
            for crops, _ in batch:
                # load it to the active device
                crops = crops.float().to(device)
                # reset the gradients back to zero
                optimizer.zero_grad()
                # compute reconstructions
                outputs = model(crops)
                # compute training reconstruction loss
                train_loss = criterion(outputs, crops)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(data_loader_train) / n_crops
        loss_history.append(loss)

        val_loss = 0
        n_crops = 1
        for batch in data_loader_test:
            n_crops = len(batch)
            for crops, _ in batch:
                crops = crops.float().to(device)
                outputs = model(crops)
                val_loss += criterion(outputs, crops).item()

        # compute the epoch training loss
        val_loss = val_loss / len(data_loader_test) / n_crops
        val_loss_history.append(val_loss)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}, val_loss = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss, val_loss))
        # save model
        torch.save(model.state_dict(), save_path + 'autoencoder_at_{}.torch'.format(epoch+1))

    return loss_history, val_loss_history


def run_weakly_supervised_classifier_training(loader_train, loader_val, model, optimizer, criterion, device,
                                              lr_scheduler=None, epochs=10, save_path='..\\res\\'):

    f_acc = Accuracy().to(device)

    acc, val_acc = 0, -1
    acc_history = []
    val_acc_history = []
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        acc = 0
        n_crops = 1
        for batch in loader_train:

            n_crops = len(batch)
            for crops, labels in batch:
                # load it to the active device
                crops = crops.float().to(device)
                labels = labels.to(device)
                # reset the gradients back to zero
                optimizer.zero_grad()

                outputs = model(crops)
                train_loss = criterion(outputs, labels)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                # add the mini-batch training acc to epoch acc
                acc += float(f_acc(outputs, labels))

        # compute epoch training loss
        loss = loss / len(loader_train) / n_crops
        # compute epoch training accuracy
        acc = acc / len(loader_train) / n_crops
        acc_history.append(acc)

        val_acc = 0
        n_crops = 1
        for batch in loader_val:
            n_crops = len(batch)
            for crops, labels in batch:
                # process drugs data
                crops = crops.float().to(device)
                labels = labels.to(device)
                outputs = model(crops)
                val_acc += float(f_acc(outputs, labels))

        # compute epoch training accuracy
        val_acc = val_acc / len(loader_val) / n_crops
        val_acc_history.append(val_acc)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}, acc = {:.4f}, val_acc = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss, acc, val_acc))
        torch.save(model.state_dict(), save_path + 'deep_classifier_at_{}.torch'.format(epoch+1))

    return acc_history, val_acc_history


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
                              device, epochs=10, ae_scheduler=None, cl_scheduler=None, save_path="..\\res\\"):

    f_acc = Accuracy().to(device)

    rec_loss_epoch = 0
    acc_epoch = 0
    rec_loss_history = []
    loss_history = []
    acc_history = []
    for epoch in range(epochs):

        start = time.time()
        ae_loss_epoch = 0
        cl_loss_epoch = 0
        rec_loss_epoch = 0
        acc_epoch = 0
        n_crops = 1
        for batch in loader_train:

            n_crops = len(batch)
            for crops, labels in batch:

                # TRAIN CLASSIFIER

                # reset gradients to zero
                cl_optimizer.zero_grad()
                # get features of drugs
                crops = crops.float().to(device)
                labels = labels.to(device)
                # retrieve encodings
                encodings = ae_model.encoder(crops)
                # run through classifier
                outputs = cl_model(encodings)
                # calculate loss
                cl_loss = cl_criterion(outputs, labels)
                cl_loss.backward()
                cl_optimizer.step()

                acc_epoch += float(f_acc(outputs, labels))

                # TRAIN AUTOENCODER

                # reset the gradients to zero
                ae_optimizer.zero_grad()
                with torch.enable_grad():

                    ae_loss = 0.
                    # compute reconstructions
                    outputs = ae_model(crops)
                    # compute training reconstruction loss
                    ae_loss += ae_criterion(outputs, crops)
                    rec_loss_epoch += ae_loss.item()
                    # add classifier loss
                    ae_loss += cl_loss.item()

                    # compute accumulated gradients
                    ae_loss.backward()
                    # perform parameter update based on current gradients
                    ae_optimizer.step()

            # add the mini-batch training loss to epoch loss
            ae_loss_epoch += ae_loss.item()
            cl_loss_epoch += cl_loss.item()

        # compute the epoch training loss
        ae_loss_epoch = ae_loss_epoch / len(loader_train) / n_crops
        cl_loss_epoch = cl_loss_epoch / len(loader_train) / n_crops
        rec_loss_epoch = rec_loss_epoch / len(loader_train) / n_crops
        acc_epoch = acc_epoch / len(loader_train) / n_crops
        rec_loss_history.append(rec_loss_epoch)
        acc_history.append(acc_epoch)
        loss_history.append(ae_loss_epoch)

        val_acc = 0
        n_crops = 1
        for batch in loader_val:

            n_crops = len(batch)
            for crops, labels in batch:
                # process drugs
                crops = crops.float().to(device)
                labels = labels.to(device)
                encodings = ae_model.encoder(crops)
                outputs = cl_model(encodings)
                val_acc += float(f_acc(outputs, labels))

        # compute epoch validation accuracy
        val_acc = val_acc / len(loader_val) / n_crops

        # update lr
        if ae_scheduler is not None:
            ae_scheduler.step()
        if cl_scheduler is not None:
            cl_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, ae_loss = {:.4f}, cl_loss = {:.4f}, rec_loss = {:.4f}, acc = {:.4f}, val_acc = {:.4f}"
              .format(epoch + 1, epochs, int((time.time() - start) / 60), ae_loss_epoch, cl_loss_epoch, rec_loss_epoch, acc_epoch, val_acc))

        torch.save(ae_model.state_dict(), save_path + 'autoencoder_at_{}.torch'.format(epoch+1))
        torch.save(cl_model.state_dict(), save_path + 'classifier_at_{}.torch'.format(epoch+1))

    return rec_loss_history, loss_history, acc_history


def plot_reconstruction(data_loader, trained_model, save_to='res/', n_images=10):

    for i in range(n_images):
        batch = next(iter(data_loader))
        crops, _ = batch[0]
        img = crops.squeeze()[0]
        img_tensor = torch.Tensor(numpy.expand_dims(crops[0], axis=0))
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


def train_autoencoder(epochs, data_loader_train, data_loader_val, trained_ae=None, device=torch.device('cuda'), run_id=""):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\ae\\{}\\'.format(run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if trained_ae is not None:
        model = trained_ae
    else:
        model = Autoencoder().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    rec_loss, val_rec_loss = run_autoencoder_training(data_loader_train, data_loader_val, model, optimizer, criterion, device,
                                                      epochs=epochs, save_path=save_path)

    # save history
    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(rec_loss))], 'loss': rec_loss, 'val_loss': val_rec_loss})
    history.to_csv(save_path + '\\history.csv', index=False)
    # save reconstruction
    plot_reconstruction(data_loader_train, model, save_to=save_path, n_images=10)


def train_deep_classifier_weakly(epochs, loader_train, loader_val, trained_cl=None, device=torch.device('cuda'), run_id=""):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\cl\\{}\\'.format(run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if trained_cl is not None:
        model = trained_cl
    else:
        model = DeepClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    acc, val_acc = run_weakly_supervised_classifier_training(loader_train, loader_val,
                                                               model, optimizer, criterion, device,
                                                               epochs=epochs,
                                                             save_path=save_path)

    # save history
    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(acc))], 'acc': acc, 'val_acc': val_acc})
    history.to_csv(save_path + '\\history.csv', index=False)


def train_together(epochs, loader_train, loader_val,
                   trained_ae=None, trained_cl=None, device=torch.device('cuda'), run_id=''):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\aecl\\{}\\'.format(run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if trained_ae is not None:
        ae = trained_ae
    else:
        ae = Autoencoder().to(device)

    ae_optimizer = optim.Adam(ae.parameters(), lr=0.0001)
    ae_criterion = nn.BCELoss()

    if trained_cl is not None:
        cl = trained_cl
    else:
        cl = Classifier().to(device)

    cl_optimizer = optim.Adam(cl.parameters(), lr=0.0001)
    cl_criterion = nn.CrossEntropyLoss()

    rec_loss, loss, acc = run_simultaneous_training(loader_train, loader_val,
                                                        ae, ae_optimizer, ae_criterion,
                                                        cl, cl_optimizer, cl_criterion,
                                                        device, epochs=epochs, save_path=save_path)

    # save history
    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(acc))], 'rec_loss': rec_loss, 'loss': loss, 'acc': acc})
    history.to_csv(save_path + '\\history.csv', index=False)

    plot_reconstruction(loader_train, ae, save_to=save_path, n_images=10)


def train_all_models(epochs, batch_size, train_dataset, test_dataset, dataset_id):

    data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print('training data:', train_dataset.__len__())
    print('validation data:', test_dataset.__len__())

    device = torch.device('cuda')

    train_autoencoder(epochs, data_loader_train, data_loader_val, device=device, run_id=dataset_id)
    train_deep_classifier_weakly(epochs, data_loader_train, data_loader_val, device=device, run_id=dataset_id)
    train_together(epochs, data_loader_train, data_loader_val, device=device, run_id=dataset_id)
    run_training_for_64x64_cuts(epochs, data_loader_train, device=device, run_id=dataset_id)


if __name__ == "__main__":

    path_to_train_data = 'D:\ETH\projects\morpho-learner\data\\train\\'
    path_to_test_data = 'D:\ETH\projects\morpho-learner\data\\test\\'
    crop_size = 64
    epochs = 50
    batch_size = 256
    train_size = -1
    test_size = -1

    print("\n\n===== NO_AUG_ONE_CROP =====\n\n")
    # make datasets with no augmentations and single crops
    train_no_aug_one_crop = MultiCropDataset(path_to_train_data, [crop_size], [1], [1], [1], no_aug=True, size_dataset=train_size)
    test_no_aug_one_crop = MultiCropDataset(path_to_test_data, [crop_size], [1], [1], [1], no_aug=True, size_dataset=test_size)
    train_all_models(epochs, batch_size, train_no_aug_one_crop, test_no_aug_one_crop, dataset_id="no_aug_one_crop")

    print("\n\n===== AUG_ONE_CROP =====\n\n")
    # make datasets with SimCLR augmentations and single crops
    train_aug_one_crop = MultiCropDataset(path_to_train_data, [crop_size], [1], [1], [1], no_aug=False, size_dataset=train_size)
    test_aug_one_crop = MultiCropDataset(path_to_test_data, [crop_size], [1], [1], [1], no_aug=False, size_dataset=test_size)
    train_all_models(epochs, batch_size, train_aug_one_crop, test_aug_one_crop, dataset_id='aug_one_crop')

    print("\n\n===== AUG_MULTI_CROP =====\n\n")
    # make datasets with SimCLR augmentations and multi-crops
    train_aug_multi_crop = MultiCropDataset(path_to_train_data, [crop_size, crop_size, crop_size], [1, 2, 2], [1, 0.5, 0.25], [1, 0.75, 0.5], no_aug=False, size_dataset=train_size)
    test_aug_multi_crop = MultiCropDataset(path_to_test_data, [crop_size, crop_size, crop_size], [1, 2, 2], [1, 0.5, 0.25], [1, 0.75, 0.5], no_aug=False, size_dataset=test_size)
    train_all_models(epochs, batch_size, train_aug_multi_crop, test_aug_multi_crop, dataset_id='aug_multi_crop')

    print("\n\n===== NO_AUG_MULTI_CROP =====\n\n")
    # make datasets with no augmentations and multi-crops
    train_no_aug_multi_crop = MultiCropDataset(path_to_train_data, [crop_size, crop_size, crop_size], [1, 2, 2], [1, 0.5, 0.25], [1, 0.75, 0.5], no_aug=True, size_dataset=train_size)
    test_no_aug_multi_crop = MultiCropDataset(path_to_test_data, [crop_size, crop_size, crop_size], [1, 2, 2], [1, 0.5, 0.25], [1, 0.75, 0.5], no_aug=True, size_dataset=test_size)
    train_all_models(epochs, batch_size, train_no_aug_multi_crop, test_no_aug_multi_crop, dataset_id='no_aug_multi_crop')
