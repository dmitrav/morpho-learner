import os, pandas, time, torch, numpy, itertools
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.io import read_image
from torchmetrics import Accuracy
from PIL import Image

from src.models import DrugClassifier, CellClassifier, Classifier
from src.analysis import get_f_transform, get_image_encodings_from_path
from src.trainer import run_supervised_classifier_training
from src import constants


def train_classifier_with_pretrained_encoder(epochs, model_name, batch_size=256):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\linear_evaluation\\'

    device = torch.device('cuda')
    transform = get_f_transform(model_name, device)

    # collect learned representations
    drugs_encodings, drugs_ids = get_image_encodings_from_path(path_to_drugs, "", transform, n=350000)
    controls_encodings, controls_ids = get_image_encodings_from_path(path_to_controls, "", transform, n=350000)
    encodings = numpy.array([*drugs_encodings, *controls_encodings])
    labels = [*[0 for x in range(len(drugs_encodings))], *[1 for x in range(len(controls_encodings))]]

    # split to train and test
    x_train, y_train, x_test, y_test = preprocess_data(encodings, labels, split_percent=0.8)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Classifier().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, model, optimizer, criterion, device,
                                                                      epochs=epochs, test_loader=test_loader)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + '{}+classifier.torch'.format(model_name))
    del drugs_encodings, controls_encodings, encodings, x_train, y_train

    return last_train_acc, last_val_acc


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


def train_linear_classifier_for_cell_line(cell_line, model, epochs,
                                          path_to_drugs='D:\ETH\projects\morpho-learner\data\cut\\',
                                          path_to_controls='D:\ETH\projects\morpho-learner\data\cut_controls\\'):

    device = torch.device('cuda')
    transform = get_f_transform(model, device)

    # collect learned representations
    drugs_encodings, drugs_ids = get_image_encodings_from_path(path_to_drugs, cell_line, transform)
    controls_encodings, controls_ids = get_image_encodings_from_path(path_to_controls, cell_line, transform, n=int(len(drugs_encodings) / 32))
    encodings = numpy.array([*drugs_encodings, *controls_encodings])
    drugs = [*drugs_ids['drugs'], *controls_ids['drugs']]

    # split to train and test
    x_train, y_train, x_test, y_test = preprocess_data(encodings, drugs, data_type='drugs', split_percent=0.8)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = DrugClassifier().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print('training for cell line {} started'.format(cell_line))
    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, model, optimizer, criterion, device,
                                                                      epochs=epochs, test_loader=test_loader)
    return last_val_acc


def train_linear_classifier_for_drug(drug, model, epochs,
                                     path_to_drugs='D:\ETH\projects\morpho-learner\data\cut\\'):

    device = torch.device('cuda')
    transform = get_f_transform(model, device)

    # collect learned representations
    drugs_encodings, drugs_ids = get_image_encodings_from_path(path_to_drugs, drug, transform)

    # split to train and test
    x_train, y_train, x_test, y_test = preprocess_data(drugs_encodings, drugs_ids['cell_lines'], data_type='cell_lines', split_percent=0.8)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    model = CellClassifier().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print('training for cell line {} started'.format(drug))
    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, model, optimizer, criterion, device,
                                                                      epochs=epochs, test_loader=test_loader)
    return last_val_acc


if __name__ == '__main__':

    # results = {}
    # for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:
    #     results[model] = []
    #     for cell_line in constants.cell_lines:
    #         val_acc = train_linear_classifier_for_cell_line(cell_line, model, 100)
    #         results[model].append(val_acc)
    #     results[model] = sum(results[model]) / len(constants.cell_lines)
    #
    # print(results)

    # results = {}
    # for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:
    #     results[model] = []
    #     for drug in constants.drugs:
    #         val_acc = train_linear_classifier_for_drug(drug, model, 200)
    #         results[model].append(val_acc)
    #     results[model] = sum(results[model]) / len(constants.cell_lines)
    #
    # print(results)

    results = {}
    for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:
        acc, val_acc = train_classifier_with_pretrained_encoder(100, model, 1024)
        results[model] = (acc, val_acc)
    print(results)