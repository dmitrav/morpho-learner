import os, pandas, time, torch, numpy, itertools
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image

from src import constants


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, label, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        if label >= 0:
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
