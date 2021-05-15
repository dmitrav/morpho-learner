import os, pandas, time, torch, numpy
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp

from src.models import Autoencoder, Classifier, DeepClassifier


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


def run_classifier_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                            model, optimizer, criterion, device, lr_scheduler=None, epochs=10):
    acc = 0
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        acc = 0
        for batch_features, batch_labels in loader_train_drugs:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            with torch.enable_grad():
                train_loss = 0

                # process drugs data
                outputs = model(batch_features)
                train_loss += criterion(outputs, batch_labels.to(device))
                true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()

                # process controls data
                batch_features, batch_labels = next(iter(loader_train_controls))
                outputs = model(batch_features.float().to(device))
                train_loss += criterion(outputs, batch_labels.to(device))
                true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()

                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

        # compute epoch training loss
        loss = loss / len(loader_train_drugs)
        # compute epoch training accuracy
        acc = acc / len(loader_train_drugs)

        val_acc = 0
        for batch_features, batch_labels in loader_val_drugs:
            # process drugs data
            batch_features = batch_features.float().to(device)
            outputs = model(batch_features)
            true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()
            # process controls data
            batch_features, batch_labels = next(iter(loader_val_controls))
            outputs = model(batch_features.float().to(device))
            true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()

            val_acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

        # compute epoch training accuracy
        val_acc = val_acc / len(loader_val_drugs)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}, acc = {:.4f}, val_acc = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss, acc, val_acc))

    return acc


def run_simultaneous_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                              ae_model, ae_optimizer, ae_criterion,
                              cl_model, cl_optimizer, cl_criterion,
                              device, epochs=10, deep_cl=True):
    rec_loss_epoch = 0
    acc_epoch = 0
    for epoch in range(epochs):

        start = time.time()
        ae_loss_epoch = 0
        cl_loss_epoch = 0
        rec_loss_epoch = 0
        acc_epoch = 0
        for batch_features, batch_labels in loader_train_drugs:

            # TRAIN CLASSIFIER

            # reset gradients to zero
            cl_optimizer.zero_grad()

            # get features of drugs
            batch_features = batch_features.float().to(device)

            # retrieve encodings
            encodings = ae_model.encoder(batch_features)
            # run through classifier
            outputs = cl_model(encodings)

            # calculate loss on drugs
            cl_loss = cl_criterion(outputs, batch_labels.to(device))
            true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()

            # get features of controls
            control_features, control_labels = next(iter(loader_train_controls))
            control_features = control_features.float().to(device)

            # retrieve encodings
            encodings = ae_model.encoder(control_features)
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
        ae_loss_epoch = ae_loss_epoch / len(loader_train_drugs)
        cl_loss_epoch = cl_loss_epoch / len(loader_train_drugs)
        rec_loss_epoch = rec_loss_epoch / len(loader_train_drugs)
        acc_epoch = acc_epoch / len(loader_train_drugs)

        val_acc = 0
        for batch_features, batch_labels in loader_val_drugs:
            # process drugs
            batch_features = batch_features.float().to(device)

            encodings = ae_model.encoder(batch_features)
            outputs = cl_model(encodings)
            true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()

            # process controls
            control_features, control_labels = next(iter(loader_train_controls))
            control_features = control_features.float().to(device)

            encodings = ae_model.encoder(control_features)
            outputs = cl_model(encodings)
            true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()

            val_acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

        # compute epoch validation accuracy
        val_acc = val_acc / len(loader_val_drugs)

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


def train_autoencoder(epochs, trained_ae=None, batch_size=256, device=torch.device('cuda')):

    path_to_data = 'D:\ETH\projects\morpho-learner\data\cut\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\ae\\'

    N = 380000  # ~90%
    training_data = CustomImageDataset(path_to_data, 0, transform=lambda x: x / 255.)
    training_data, test_data = torch.utils.data.random_split(training_data, [N, training_data.__len__() - N])

    data_loader_train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if trained_ae is not None:
        model = trained_ae
    else:
        model = Autoencoder().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 35], gamma=0.5)
    criterion = nn.BCELoss()

    # best loss = 0.6331
    last_rec_loss = run_autoencoder_training(data_loader_train, data_loader_test, model, optimizer, criterion, device, epochs=epochs)

    save_path = save_path.replace('ae', 'ae_{}'.format(round(last_rec_loss, 4)))
    plot_reconstruction(data_loader_train, model, save_to=save_path, n_images=50)

    torch.save(model.state_dict(), save_path + 'autoencoder.torch')


def train_classifier_with_pretrained_encoder(epochs, trained_ae, batch_size=256, deep=False):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\cl\\'

    device = torch.device('cuda')

    N = 380000  # ~90%
    transform = lambda x: trained_ae.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=transform)
    training_drugs, the_rest = torch.utils.data.random_split(training_drugs, [N, training_drugs.__len__() - N])
    validation_drugs, _ = torch.utils.data.random_split(the_rest, [N // 2, the_rest.__len__() - N // 2])

    training_controls = CustomImageDataset(path_to_controls, 1, transform=transform)
    training_controls, the_rest = torch.utils.data.random_split(training_controls, [N, training_controls.__len__() - N])
    validation_controls, _ = torch.utils.data.random_split(the_rest, [N // 2, the_rest.__len__() - N // 2])

    loader_train_drugs = DataLoader(training_drugs, batch_size=batch_size, shuffle=True)
    loader_train_controls = DataLoader(training_controls, batch_size=batch_size, shuffle=True)
    loader_val_drugs = DataLoader(validation_drugs, batch_size=batch_size, shuffle=True)
    loader_val_controls = DataLoader(validation_controls, batch_size=batch_size, shuffle=True)

    if deep:
        model = DeepClassifier().to(device)
    else:
        model = Classifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    last_epoch_acc = run_classifier_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                                             model, optimizer, criterion, device, epochs=epochs)
    if deep:
        save_path = save_path.replace('cl', 'dcl_{}'.format(round(last_epoch_acc, 4)))
    else:
        save_path = save_path.replace('cl', 'cl_{}'.format(round(last_epoch_acc, 4)))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + 'classifier.torch')


def train_deep_classifier_alone(epochs, trained_cl=None, batch_size=256, device=torch.device('cuda')):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\dcl\\'

    Nd, Nc = 380000, 330000  # ~89%
    transform = lambda x: x / 255.

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=transform)
    training_drugs, the_rest = torch.utils.data.random_split(training_drugs, [Nd, training_drugs.__len__() - Nd])
    validation_drugs, _ = torch.utils.data.random_split(the_rest, [Nd // 2, the_rest.__len__() - Nd // 2])

    training_controls = CustomImageDataset(path_to_controls, 1, transform=transform)
    training_controls, the_rest = torch.utils.data.random_split(training_controls, [Nc, training_controls.__len__() - Nc])
    validation_controls, _ = torch.utils.data.random_split(the_rest, [Nc // 2, the_rest.__len__() - Nc // 2])

    loader_train_drugs = DataLoader(training_drugs, batch_size=batch_size, shuffle=True)
    loader_train_controls = DataLoader(training_controls, batch_size=batch_size, shuffle=True)
    loader_val_drugs = DataLoader(validation_drugs, batch_size=batch_size, shuffle=True)
    loader_val_controls = DataLoader(validation_controls, batch_size=batch_size, shuffle=True)

    if trained_cl is not None:
        model = trained_cl
    else:
        model = DeepClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00045)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    last_epoch_acc = run_classifier_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                                             model, optimizer, criterion, device, epochs=epochs)

    save_path = save_path.replace('dcl', 'dcl_{}'.format(round(last_epoch_acc, 4)))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + 'deep_classifier.torch')


def train_together(epochs, trained_ae=None, trained_cl=None, batch_size=256, device=torch.device('cuda')):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\aecl\\'

    if trained_ae is not None:
        ae = trained_ae
    else:
        ae = Autoencoder().to(device)

    ae_optimizer = optim.Adam(ae.parameters(), lr=0.00015)
    ae_criterion = nn.BCELoss()

    if trained_cl is not None:
        cl = trained_cl
    else:
        cl = Classifier().to(device)

    cl_optimizer = optim.Adam(cl.parameters(), lr=0.00045)
    cl_criterion = nn.CrossEntropyLoss()

    Nd, Nc = 380000, 330000  # ~89%

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)
    training_drugs, validation_drugs = torch.utils.data.random_split(training_drugs, [Nd, training_drugs.__len__() - Nd])

    training_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)
    training_controls, the_rest = torch.utils.data.random_split(training_controls, [Nc, training_controls.__len__() - Nc])
    validation_controls, _ = torch.utils.data.random_split(the_rest, [len(validation_drugs), the_rest.__len__() - len(validation_drugs)])

    print('total drugs:', training_drugs.__len__())
    print('total controls:', training_controls.__len__(), '\n')

    loader_train_drugs = DataLoader(training_drugs, batch_size=batch_size, shuffle=True)
    loader_train_controls = DataLoader(training_controls, batch_size=batch_size, shuffle=True)
    loader_val_drugs = DataLoader(validation_drugs, batch_size=batch_size, shuffle=True)
    loader_val_controls = DataLoader(validation_controls, batch_size=batch_size, shuffle=True)

    last_rec_loss, last_acc = run_simultaneous_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                                                        ae, ae_optimizer, ae_criterion,
                                                        cl, cl_optimizer, cl_criterion,
                                                        device, epochs=epochs)

    save_path = save_path.replace('aecl', 'aecl_{}_{}'.format(round(last_rec_loss, 4), round(last_acc, 4)))

    plot_reconstruction(loader_train_drugs, ae, save_to=save_path, n_images=30)
    torch.save(ae.state_dict(), save_path + 'ae.torch')
    torch.save(cl.state_dict(), save_path + 'cl.torch')


if __name__ == "__main__":

    train_ae_alone = False
    train_cl_alone = True
    train_both_simultaneously = True
    train_cl_alone_with_pretrained_ae = False

    device = torch.device('cuda')

    if train_ae_alone:
        # load model and continue training
        path_to_ae_model = "D:\ETH\projects\morpho-learner\\res\\ae_at_60_0.6673\\autoencoder.torch"
        ae = Autoencoder().to(device)
        ae.load_state_dict(torch.load(path_to_ae_model, map_location=device))
        ae.eval()

        train_autoencoder(40, trained_ae=ae, device=device)

    if train_cl_alone:
        # # load model and continue training
        # path_to_cl_model = "D:\ETH\projects\morpho-learner\\res\\dcl_at_60_0.7159\\deep_classifier.torch"
        # cl = DeepClassifier().to(device)
        # cl.load_state_dict(torch.load(path_to_cl_model, map_location=device))
        # cl.eval()

        train_deep_classifier_alone(60, device=device)

    if train_both_simultaneously:

        path_to_ae_model = "D:\ETH\projects\morpho-learner\\res\\aecl_at_60_0.6672_0.7217\\ae.torch"
        ae = Autoencoder().to(device)
        ae.load_state_dict(torch.load(path_to_ae_model, map_location=device))
        ae.eval()

        path_to_cl_model = "D:\ETH\projects\morpho-learner\\res\\aecl_at_60_0.6672_0.7217\\dcl.torch"
        cl = Classifier().to(device)
        cl.load_state_dict(torch.load(path_to_cl_model, map_location=device))
        cl.eval()

        train_together(40, device=device)

    if train_cl_alone_with_pretrained_ae:
        """ classification based on the reduced dimensions, obtained by pretrained autoencoder
            not directly comparable to autoencoder and advesarial approaches -> probably redundant """

        path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\ae_0.6673\\'
        # load trained autoencoder to use it in the transform
        ae = Autoencoder().to(device)
        ae.load_state_dict(torch.load(path_to_ae_model+'autoencoder.torch', map_location=device))
        ae.eval()

        train_classifier_with_pretrained_encoder(100, ae)
