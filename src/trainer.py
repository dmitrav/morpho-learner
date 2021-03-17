import os, pandas, time, torch, numpy
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


def train_model(data_loader, model, optimizer, criterion, device, lr_scheduler=None, epochs=5):

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
            print('lr={}'.format(optimizer.param_groups[0]['lr']))  # debug
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} sec, loss = {:.4f}".format(epoch + 1, epochs, int(time.time() - start), loss))

    return model


def plot_reconstruction(data_loader, trained_model, n_images=10):

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
        pyplot.show()


if __name__ == "__main__":

    path_to_data = '/Users/{}/ETH/projects/morpho-learner/data/cut/'.format(user)
    training_data = CustomImageDataset(path_to_data, transform=lambda x: x / 255.)
    # training_data, test_data = torch.utils.data.random_split(training_data, [10000, 70000])
    training_data, test_data = torch.utils.data.random_split(training_data, [5000, 75000])

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    model = Autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5], gamma=0.5)
    criterion = nn.BCELoss()
    device = device("cpu")

    # best loss = 0.6331
    trained_model = train_model(train_dataloader, model, optimizer, criterion, device, lr_scheduler=scheduler, epochs=10)
    plot_reconstruction(train_dataloader, trained_model, n_images=30)

    # TODO:
    #  - try torch.multiprocessing
    #  - train and save model to use encoder later
