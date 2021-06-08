
from byol_pytorch import BYOL
from torchvision import models

import os, pandas, time, torch, numpy, uuid, seaborn, random, shutil, traceback
from PIL import Image
from matplotlib import pyplot
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp
from torch.nn import Sequential
from vit_pytorch import ViT, Dino
from vit_pytorch.recorder import Recorder

from src.models import Autoencoder, Classifier, DeepClassifier
from src.trainer import CustomImageDataset, JointImageDataset
from src.constants import vit_par_types
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier_with_pretrained_encoder
from src.trainer import train_deep_classifier_weakly
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering


def get_byol_pars(im_size, randomize=True):

    if randomize:
        return dict(image_size=im_size,  # 256
                    hidden_layer='model.9',
                    projection_size=random.sample([64, 128, 256, 512, 1024, 2048], 1)[0],  # 256
                    projection_hidden_size=random.sample([512, 1024, 2048, 4096, 8192], 1)[0],  # 4096
                    augment_fn=None,
                    augment_fn2=None,
                    moving_average_decay=random.sample([0.8, 0.9, 0.99], 1)[0],
                    use_momentum=True)
    else:
        return dict(image_size=im_size, hidden_layer='model.9')


def generate_grid(grid_size, image_size, randomize=True):

    grid = {'id': [], 'byol': []}
    for _ in range(grid_size):
        grid['id'].append(str(uuid.uuid4())[:8])
        grid['byol'].append(get_byol_pars(image_size, randomize=randomize))

    return grid


def save_history_and_parameters(loss_history, byol_pars, save_path):

    # save history
    history = pandas.DataFrame({'epoch': [x+1 for x in range(len(loss_history))], 'loss': loss_history})
    history.to_csv(save_path + '\\history.csv', index=False)

    # plot history
    seaborn.lineplot(data=history, x='epoch', y='loss')
    pyplot.grid()
    pyplot.savefig(save_path + '\\loss.png')
    pyplot.close()

    # save vit parameters
    pandas.DataFrame(byol_pars, index=['values'], columns=byol_pars.keys()).T \
        .to_csv(save_path + '\\byol_pars.csv', index=True)

    print('history and parameters saved\n')


def run_training_for_64x64_cuts(model, epochs, device, grid=None, save_path='D:\ETH\projects\morpho-learner\\res\\byol\\'):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'

    batch_size = 512
    N = 350000

    if grid is None:
        grid_size = 1
        grid = generate_grid(grid_size, 64, randomize=True)

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)  # ~429000
    training_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)  # ~370000
    training_drugs, _ = torch.utils.data.random_split(training_drugs, [N, training_drugs.__len__() - N])
    training_controls, _ = torch.utils.data.random_split(training_controls, [N, training_controls.__len__() - N])

    joint_data = JointImageDataset([training_drugs, training_controls], transform=lambda x: x / 255.)
    data_loader = DataLoader(joint_data, batch_size=batch_size, shuffle=True)

    train(model, grid, epochs, data_loader, device, save_path)


def train(model, grid, epochs, data_loader, device, save_path):

    for i, id in enumerate(grid['id']):

        print(pandas.DataFrame(grid['byol'][i], index=['values'], columns=grid['byol'][i].keys()).T.to_string())
        print('training for set {} started'.format(id))
        if not os.path.exists(save_path + id):
            os.makedirs(save_path + id)

        try:
            learner = BYOL(model, **grid['byol'][i]).to(device)
            opt = torch.optim.Adam(learner.parameters(), lr=0.0001)

            loss_history = []
            try:
                for epoch in range(epochs):
                    start = time.time()
                    epoch_loss = 0
                    for batch_features in data_loader:
                        images = batch_features[0].float().to(device)
                        loss = learner(images)
                        epoch_loss += loss.item()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

                    epoch_loss = epoch_loss / len(data_loader)
                    loss_history.append(epoch_loss)
                    print("epoch {}: {} min, loss = {:.4f}".format(epoch + 1, int((time.time() - start) / 60),
                                                                   epoch_loss))
                    # save network
                    torch.save(model.state_dict(), save_path + id + '\\dcl+byol_at_{}.torch'.format(epoch))

                    if epoch >= 2:
                        if epoch_loss > loss_history[epoch - 1] > loss_history[epoch - 2] or epoch_loss > loss_history[0]:
                            # if loss grows, stop training
                            break
                        elif round(epoch_loss, 4) == round(loss_history[epoch - 1], 4):
                            # if loss doesn't fall, stop
                            break

                print('{}/{} completed'.format(i + 1, len(grid['id'])))
                save_history_and_parameters(loss_history, grid['byol'][i], save_path + id)

            except Exception as e:
                print('{}/{} failed with {}\n'.format(i + 1, len(grid['id']), e))
                shutil.rmtree(save_path + id)
        except Exception as e:
            print('{}/{} failed building byol with {}\n'.format(i + 1, len(grid['id']), e))
            print(traceback.print_exc())
            shutil.rmtree(save_path + id)


def get_image_tensor(path):
    image = numpy.array(Image.open(path).convert('RGB'))  # 3 channels
    image = numpy.moveaxis(image, -1, 0)  # set channels as the first dim
    image = torch.Tensor(image)  # make tensor
    image = image / 255.  # transform
    image = torch.unsqueeze(image, 0)  # add batch dim

    return image


def get_accuracy(model,
                 path_to_drugs='D:\ETH\projects\morpho-learner\data\cut\\',
                 path_to_controls='D:\ETH\projects\morpho-learner\data\cut_controls\\',
                 device=torch.device('cuda')):

    Nd, Nc = 380000, 330000  # ~89%
    transform = lambda x: x / 255.
    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=transform)
    training_drugs, _ = torch.utils.data.random_split(training_drugs, [Nd, training_drugs.__len__() - Nd])
    training_controls = CustomImageDataset(path_to_controls, 1, transform=transform)
    training_controls, _ = torch.utils.data.random_split(training_controls, [Nc, training_controls.__len__() - Nc])

    loader_train_drugs = DataLoader(training_drugs, batch_size=512, shuffle=True)
    loader_train_controls = DataLoader(training_controls, batch_size=512, shuffle=True)

    acc = 0
    for batch_features, batch_labels in loader_train_drugs:
        # process drugs data
        batch_features = batch_features.float().to(device)
        outputs = model(batch_features)
        true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()
        # process controls data
        batch_features, batch_labels = next(iter(loader_train_controls))
        outputs = model(batch_features.float().to(device))
        true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()

        acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

    # compute epoch training accuracy
    acc = round(acc / len(loader_train_drugs), 4)
    print('acc={}'.format(acc))
    return acc


if __name__ == '__main__':
    pass

    # TODO:
    #  - try weight decay,
    #  - try Lars optimizer and full BYOL paper set-up,
    #  - train classifier on top of learned representations,
    #  - repeat the analysis for learned representations
