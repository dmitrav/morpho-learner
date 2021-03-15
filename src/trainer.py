import os, pandas, time, torch
from matplotlib import pyplot
from torch import nn, optim, device
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from src.constants import user
from src.models import Autoencoder


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = pandas.DataFrame({'img': [f for f in os.listdir(img_dir)],
                                            'label': [0 for f in os.listdir(img_dir)]})

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


if __name__ == "__main__":

    path_to_data = '/Users/{}/ETH/projects/morpho-learner/data/cut/'.format(user)
    training_data = CustomImageDataset(path_to_data)
    training_data, test_data = torch.utils.data.random_split(training_data, [10000, 70000])
    train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)

    model = Autoencoder()
    print(model)

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    device = device("cpu")

    epochs = 10
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        for batch_features in train_dataloader:
            # reshape mini-batch data to [n_batches, n_features] matrix
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
        loss = loss / len(train_dataloader)

        # display the epoch training loss
        print("epoch {}/{}: {}s, loss = {:.4f}".format(epoch + 1, epochs, time.time() - start, loss))

