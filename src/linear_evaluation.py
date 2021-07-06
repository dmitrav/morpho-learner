import os, pandas, time, torch, numpy, itertools
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.io import read_image
from torchmetrics import Accuracy
from PIL import Image

from src.models import DrugClassifier, CellClassifier
from src.analysis import get_f_transform, get_image_encodings_from_path
from src.trainer import run_supervised_classifier_training
from src import constants


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, label, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        if label > 0:
            self.img_labels = pandas.DataFrame({
                'img': [f for f in os.listdir(img_dir)],
                'label': [label for f in os.listdir(img_dir)]
            })
        else:
            self.img_labels = pandas.DataFrame({
                'img': [f for f in os.listdir(img_dir)],
                'label': [self._infer_label(f) for f in os.listdir(img_dir)]
            })

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

    def _infer_label(self, filename):
        """ This method infers cell lines labels from the filenames. """

        mapper = dict(zip(constants.cell_lines, [x for x in range(len(constants.cell_lines))]))
        self.n_classes = len(constants.cell_lines)

        for line in constants.cell_lines:
            if line in filename:
                return mapper[line]


class JointImageDataset(Dataset):
    def __init__(self, datasets, transform=None, target_transform=None, n_channels=1):

        for subset in datasets:
            if subset.dataset.img_labels.columns[0] == 'path':
                # some weird bug made me do this
                continue
            else:
                subset.dataset.img_labels.insert(0, 'path', subset.dataset.img_dir)

        self.img_labels = pandas.concat([subset.dataset.img_labels for subset in datasets])
        self.transform = transform
        self.target_transform = target_transform
        self.n_channels = n_channels
        self.n_classes = self.img_labels['label'].max()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 1])

        if self.n_channels == 3:
            # read 3 channels
            image = numpy.array(Image.open(img_path).convert('RGB'))
            image = numpy.moveaxis(image, -1, 0)  # set channels as the first dim
            image = torch.Tensor(image)
        elif self.n_channels == 1:
            image = read_image(img_path)
        else:
            raise ValueError()

        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample


def train_classifier_with_pretrained_encoder(epochs, model, batch_size=256):
    """ Doesn't really work... """

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\linear_evaluation\\'

    device = torch.device('cuda')

    transform = get_f_transform(model, device)
    # Nd, Nc, Ncv = 380000, 20000, 1500  # ~89% of drugs
    Nd, Nc = 350000, 350000

    training_drugs = CustomImageDataset(path_to_drugs, -1, transform=transform)
    training_drugs, the_rest = torch.utils.data.random_split(training_drugs, [Nd, training_drugs.__len__() - Nd])
    validation_drugs, _ = torch.utils.data.random_split(the_rest, [the_rest.__len__(), 0])

    training_controls = CustomImageDataset(path_to_controls, -1, transform=transform)
    training_controls, the_rest = torch.utils.data.random_split(training_controls, [Nc, training_controls.__len__() - Nc])
    validation_controls, _ = torch.utils.data.random_split(the_rest, [the_rest.__len__(), 0])

    training_data = JointImageDataset([training_drugs, training_controls], transform=transform)
    validation_data = JointImageDataset([validation_drugs.dataset, validation_controls.dataset], transform=transform)

    loader_train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    model = CellClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    last_train_acc, last_val_acc = run_supervised_classifier_training(loader_train, loader_val, model, optimizer,
                                                                      criterion, device, epochs=epochs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + '{}+linear.torch'.format(model))


def preprocess_data(encodings, ids, data_type='drugs'):

    if data_type == 'drugs':
        mapper = dict(zip(constants.drugs, [x for x in range(len(constants.drugs))]))
        labels = [mapper[drug] for drug in ids]
    elif data_type == 'cell_lines':
        mapper = dict(zip(constants.cell_lines, [x for x in range(len(constants.cell_lines))]))
        labels = [mapper[cell_line] for cell_line in ids]
    else:
        raise ValueError('Unknown type.')

    data = pandas.DataFrame(encodings)
    data.insert(0, 'label', labels)
    data = data.sample(frac=1)  # shuffle

    data_train = data.iloc[:int(data.shape[0] * 0.8), :]
    data_test = data.iloc[int(data.shape[0] * 0.8):, :]

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
    x_train, y_train, x_test, y_test = preprocess_data(encodings, drugs, data_type='drugs')

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = DrugClassifier().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print('training for cell line {} started'.format(cell_line))
    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, test_loader, model, optimizer,
                                                                      criterion, device, epochs=epochs)
    return last_val_acc


def train_linear_classifier_for_drug(drug, model, epochs,
                                     path_to_drugs='D:\ETH\projects\morpho-learner\data\cut\\'):

    device = torch.device('cuda')
    transform = get_f_transform(model, device)

    # collect learned representations
    drugs_encodings, drugs_ids = get_image_encodings_from_path(path_to_drugs, drug, transform)

    # split to train and test
    x_train, y_train, x_test, y_test = preprocess_data(drugs_encodings, drugs_ids['cell_lines'], data_type='cell_lines')

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    model = CellClassifier().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print('training for cell line {} started'.format(drug))
    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, test_loader, model, optimizer,
                                                                      criterion, device, epochs=epochs)
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

    results = {}
    for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:
        results[model] = []
        for drug in constants.drugs:
            val_acc = train_linear_classifier_for_drug(drug, model, 200)
            results[model].append(val_acc)
        results[model] = sum(results[model]) / len(constants.cell_lines)

    print(results)