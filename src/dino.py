
import os, pandas, time, torch, numpy, uuid, seaborn, random, shutil
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp
from torch.nn import Sequential
from vit_pytorch import ViT, Dino

from src.models import Autoencoder, Classifier, DeepClassifier
from src.trainer import CustomImageDataset, JointImageDataset
from src.trainer import plot_reconstruction, train_together, train_autoencoder, train_classifier_with_pretrained_encoder
from src.trainer import train_deep_classifier_alone
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering


def get_vit_pars(randomize=True):

    if randomize:
        return dict(image_size=64,  # 256
                    patch_size=random.sample([4, 8, 16], 1)[0],  # 32
                    num_classes=random.sample([2, 21, 32, 100, 1000], 1)[0],  # 1000
                    dim=random.sample([64, 128, 256, 512], 1)[0],  # 1024
                    depth=random.sample([x for x in range(1, 7)], 1)[0],  # 6
                    heads=random.sample([x for x in range(1, 17)], 1)[0],  # 16
                    mlp_dim=random.sample([64, 128, 256, 512, 1024], 1)[0],  # 2048
                    dropout=random.sample([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1)[0],
                    emb_dropout=random.sample([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1)[0])
    else:
        return dict(image_size=64, patch_size=8, num_classes=2, dim=128, depth=2, heads=3, mlp_dim=256)


def get_dino_pars(randomize=False):

    if randomize:
        return dict(image_size=64,  # 256
                    hidden_layer='to_latent',
                    projection_hidden_size=64,  # 256
                    projection_layers=random.sample([1, 2, 3, 4], 1)[0],  # 4
                    num_classes_K=random.sample([1024, 2048, 4096, 8192, 16334, 65336], 1)[0],  # 65336
                    student_temp=random.sample([0.5, 0.6, 0.7, 0.8, 0.9], 1)[0],  # 0.9
                    teacher_temp=random.sample([0.02, 0.04, 0.05, 0.06, 0.07, 0.09], 1)[0],  # 0.04-0.07
                    local_upper_crop_scale=random.sample([0.2, 0.3, 0.4, 0.5, 0.6], 1)[0],  # 0.4
                    global_lower_crop_scale=random.sample([0.2, 0.3, 0.4, 0.5, 0.6], 1)[0],  # 0.5
                    moving_average_decay=random.sample([0.5, 0.6, 0.7, 0.8, 0.9], 1)[0],  # 0.9-0.999
                    center_moving_average_decay=random.sample([0.5, 0.6, 0.7, 0.8, 0.9], 1)[0])  # 0.9-0.999
    else:
        return dict(image_size=64, hidden_layer='to_latent', projection_hidden_size=64, projection_layers=4,
                    num_classes_K=16334, student_temp=0.9, teacher_temp=0.04, local_upper_crop_scale=0.4,
                    global_lower_crop_scale=0.5, moving_average_decay=0.9, center_moving_average_decay=0.9)


def generate_grid(grid_size, random_dino=False):

    grid = {'id': [], 'vit': [], 'dino': []}
    for _ in range(grid_size):
        grid['id'].append(str(uuid.uuid4())[:8])
        grid['vit'].append(get_vit_pars(randomize=True))
        grid['dino'].append(get_dino_pars(randomize=random_dino))

    return grid


def save_history_and_parameters(grid, loss_history, save_path):

    # save history
    history = pandas.DataFrame({'epoch': [x for x in range(1, i + 2)], 'loss': loss_history})
    history.to_csv(save_path + id + '\\history.csv', index=False)

    # plot history
    seaborn.lineplot(data=history, x='epoch', y='loss')
    pyplot.savefig(save_path + id + '\\loss.png')
    pyplot.close()

    # save vit parameters
    pandas.DataFrame(grid['vit'][i], index=['pars']).T \
        .to_csv(save_path + id + '\\vit_pars.csv', index=False)

    # save dino parameters
    pandas.DataFrame(grid['dino'][i], index=['pars']).T \
        .to_csv(save_path + id + '\\dino_pars.csv', index=False)

    print('history and parameters saved')


if __name__ == "__main__":

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\dino\\'

    grid_size = 100
    batch_size = 512
    N = 300000
    epochs = 10

    grid = generate_grid(grid_size, random_dino=False)

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)
    training_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)
    training_drugs, _ = torch.utils.data.random_split(training_drugs, [N, training_drugs.__len__() - N])
    training_controls, _ = torch.utils.data.random_split(training_controls, [N, training_controls.__len__() - N])

    joint_data = JointImageDataset([training_drugs, training_controls], transform=lambda x: x / 255.)
    data_loader = DataLoader(joint_data, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda')
    for i, id in enumerate(grid['id']):

        print('training for set {} started'.format(id))
        if not os.path.exists(save_path + id):
            os.makedirs(save_path + id)

        model = ViT(**grid['vit'][i]).to(device)
        learner = Dino(model, **grid['dino'][i]).to(device)

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
                print("epoch {}: {} min, loss = {:.4f}".format(epoch+1, int((time.time() - start) / 60), epoch_loss))

                # save network
                torch.save(model.state_dict(), save_path + id + '\\ViT_at_{}.torch'.format(epoch))

                if epoch >= 2:
                    if epoch_loss > loss_history[epoch-1] > loss_history[epoch-2]:
                        # if loss grows, stop training
                        break

            print('{}/{} completed\n'.format(i+1, grid_size))
            save_history_and_parameters(grid, loss_history, save_path + id)

        except Exception as e:
            print('{}/{} failed with {}\n'.format(i+1, grid_size, e))
            shutil.rmtree(save_path + id)



