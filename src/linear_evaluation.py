import os, pandas, time, torch, numpy, itertools
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchmetrics import Accuracy
from PIL import Image

from src.models import DrugClassifier
from src.analysis import get_f_transform
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
        """ This method infers drugs labels from the filenames. """

        mapper = dict(zip(constants.drugs, [x for x in range(len(constants.drugs))]))
        self.n_classes = len(constants.drugs)

        for drug in constants.drugs:
            if drug in filename:
                return mapper[drug]


def train_classifier_with_pretrained_encoder(epochs, model, batch_size=256):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\linear_evaluation\\'

    device = torch.device('cuda')

    transform = get_f_transform(model, device)
    Nd, Nc = 380000, 330000  # ~89% of drugs

    training_drugs = CustomImageDataset(path_to_drugs, -1, transform=transform)
    training_drugs, the_rest = torch.utils.data.random_split(training_drugs, [Nd, training_drugs.__len__() - Nd])
    validation_drugs, _ = torch.utils.data.random_split(the_rest, [the_rest.__len__(), 0])

    training_controls = CustomImageDataset(path_to_controls, -1, transform=transform)
    training_controls, the_rest = torch.utils.data.random_split(training_controls, [Nc, training_controls.__len__() - Nc])
    validation_controls, _ = torch.utils.data.random_split(the_rest, [the_rest.__len__(), 0])

    loader_train_drugs = DataLoader(training_drugs, batch_size=batch_size, shuffle=True)
    loader_val_drugs = DataLoader(validation_drugs, batch_size=batch_size, shuffle=True)
    loader_train_controls = DataLoader(training_controls, batch_size=batch_size // 32, shuffle=True)
    loader_val_controls = DataLoader(validation_controls, batch_size=batch_size // 32, shuffle=True)

    model = DrugClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    last_epoch_acc = run_supervised_classifier_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                                                        model, optimizer, criterion, device, epochs=epochs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + '{}+linear.torch'.format(model))


if __name__ == '__main__':

    train_classifier_with_pretrained_encoder(10, 'unsupervised')