import os, pandas, time, torch, numpy, itertools
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.io import read_image
from torchmetrics import Accuracy
from PIL import Image

from src.models import Classifier
from src.analysis import get_f_transform, get_image_encodings_from_path
from src.trainer import run_supervised_classifier_training
from src import constants


def train_classifier_with_pretrained_encoder(epochs, model_name, setting_name, batch_size=256):

    path_to_drugs_train = 'D:\ETH\projects\morpho-learner\data\\train\\drugs\\'
    path_to_controls_train = 'D:\ETH\projects\morpho-learner\data\\train\\controls\\'
    path_to_drugs_test = 'D:\ETH\projects\morpho-learner\data\\test\\drugs\\'
    path_to_controls_test = 'D:\ETH\projects\morpho-learner\data\\test\\controls\\'

    save_path = 'D:\ETH\projects\morpho-learner\\res\\linear_evaluation\\{}\\{}\\'.format(model_name, setting_name)

    device = torch.device('cuda')
    transform = get_f_transform(model_name, setting_name, device)

    # collect learned representations
    drugs_train, _ = get_image_encodings_from_path(path_to_drugs_train, "", transform)
    controls_train, _ = get_image_encodings_from_path(path_to_controls_train, "", transform)
    drugs_test, _ = get_image_encodings_from_path(path_to_drugs_test, "", transform)
    controls_test, _ = get_image_encodings_from_path(path_to_controls_test, "", transform)

    print("train set: {} drugs, {} controls".format(len(drugs_train), len(controls_train)))
    print("test set: {} drugs, {} controls".format(len(drugs_test), len(controls_test)))

    x_train = [*drugs_train, *controls_train]
    y_train = [*[1 for x in range(len(drugs_train))], *[0 for x in range(len(controls_train))]]
    x_test = [*drugs_test, *controls_test]
    y_test = [*[1 for x in range(len(drugs_test))], *[0 for x in range(len(controls_test))]]

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Classifier().to(device)

    lrs = [0.1, 0.05, 0.01]
    ms = [0.9, 0.8, 0.7]
    wds = [1e-3, 1e-4, 1e-5]

    for lr in lrs:
        for m in ms:
            for wd in wds:

                params = 'lr={},m={},wd={}'.format(lr, m, wd)
                if not os.path.exists(save_path + '\\' + params + '\\'):
                    os.makedirs(save_path + '\\' + params + '\\')

                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
                criterion = nn.CrossEntropyLoss()

                last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, model, optimizer, criterion, device,
                                                                                  epochs=epochs, test_loader=test_loader,
                                                                                  save_to=save_path + '\\' + params + '\\')


def preprocess_data(encodings, ids, data_type='', split_percent=None):

    if data_type == 'drugs':
        mapper = dict(zip(constants.drugs, [x for x in range(len(constants.drugs))]))
        labels = [mapper[drug] for drug in ids]
    elif data_type == 'cell_lines':
        mapper = dict(zip(constants.cell_lines, [x for x in range(len(constants.cell_lines))]))
        labels = [mapper[cell_line] for cell_line in ids]
    else:
        labels = ids

    data = pandas.DataFrame(encodings)
    data.insert(0, 'label', labels)
    data = data.sample(frac=1)  # shuffle

    if split_percent is None:
        # do not split
        x_train = data.iloc[:, 1:].values
        y_train = data.iloc[:, 0].values
        x_test = None
        y_test = None
    else:
        data_train = data.iloc[:int(data.shape[0] * split_percent), :]
        data_test = data.iloc[int(data.shape[0] * split_percent):, :]

        x_train = data_train.iloc[:, 1:].values
        x_test = data_test.iloc[:, 1:].values
        y_train = data_train.iloc[:, 0].values
        y_test = data_test.iloc[:, 0].values

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

            train_classifier_with_pretrained_encoder(25, model, setting, batch_size=1024)