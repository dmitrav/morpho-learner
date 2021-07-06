
import os, pandas, time, torch, numpy, uuid, seaborn, random, shutil
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
from src.trainer import plot_reconstruction, train_together, train_autoencoder
from src.trainer import train_deep_classifier_weakly
from src.analysis import plot_drugs_clustering, plot_cell_lines_clustering


def get_vit_pars(im_size, randomize=True):

    if randomize:
        return dict(image_size=im_size,  # 256
                    patch_size=random.sample([4, 8, 16, 32], 1)[0],  # 32
                    num_classes=random.sample([x for x in range(2, 11)], 1)[0],  # 1000
                    dim=random.sample([64, 128, 256, 512, 1024], 1)[0],  # 1024
                    depth=random.sample([x for x in range(1, 7)], 1)[0],  # 6
                    heads=random.sample([x for x in range(1, 17)], 1)[0],  # 16
                    mlp_dim=random.sample([64, 128, 256, 512, 1024, 2048], 1)[0],  # 2048
                    dropout=random.sample([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1)[0],
                    emb_dropout=random.sample([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1)[0])
    else:
        # return dict(image_size=64, patch_size=8, num_classes=2, dim=128, depth=2, heads=3, mlp_dim=256)
        return dict(image_size=256, patch_size=32, num_classes=1000, dim=1024, depth=6, heads=16, mlp_dim=2048)


def get_dino_pars(im_size, randomize=False):

    if randomize:
        return dict(image_size=im_size,  # 256
                    hidden_layer='to_latent',
                    projection_hidden_size=im_size,  # 256
                    projection_layers=random.sample([1, 2, 3, 4], 1)[0],  # 4
                    num_classes_K=random.sample([4096, 8192, 16334, 65336], 1)[0],  # 65336
                    student_temp=random.sample([0.7, 0.8, 0.9], 1)[0],  # 0.9
                    teacher_temp=random.sample([0.02, 0.04, 0.05, 0.07, 0.09], 1)[0],  # 0.04-0.07
                    local_upper_crop_scale=random.sample([0.3, 0.4, 0.5], 1)[0],  # 0.4
                    global_lower_crop_scale=random.sample([0.4, 0.5, 0.6], 1)[0],  # 0.5
                    moving_average_decay=random.sample([0.8, 0.9, 0.99], 1)[0],  # 0.9-0.999
                    center_moving_average_decay=random.sample([0.8, 0.9, 0.99], 1)[0])  # 0.9-0.999
    else:
        # return dict(image_size=64, hidden_layer='to_latent', projection_hidden_size=64, projection_layers=4,
        #             num_classes_K=16334, student_temp=0.9, teacher_temp=0.04, local_upper_crop_scale=0.4,
        #             global_lower_crop_scale=0.5, moving_average_decay=0.9, center_moving_average_decay=0.9)
        return dict(image_size=256, hidden_layer='to_latent', projection_hidden_size=256, projection_layers=4,
                    num_classes_K=65336, student_temp=0.9, teacher_temp=0.04, local_upper_crop_scale=0.4,
                    global_lower_crop_scale=0.5, moving_average_decay=0.9, center_moving_average_decay=0.9)


def generate_grid(grid_size, image_size, random_vit=True, random_dino=False):

    grid = {'id': [], 'vit': [], 'dino': []}
    for _ in range(grid_size):
        grid['id'].append(str(uuid.uuid4())[:8])
        grid['vit'].append(get_vit_pars(image_size, randomize=random_vit))
        grid['dino'].append(get_dino_pars(image_size, randomize=random_dino))

    return grid


def save_history_and_parameters(loss_history, vit_pars, dino_pars, save_path):

    # save history
    history = pandas.DataFrame({'epoch': [x+1 for x in range(len(loss_history))], 'loss': loss_history})
    history.to_csv(save_path + '\\history.csv', index=False)

    # plot history
    seaborn.lineplot(data=history, x='epoch', y='loss')
    pyplot.grid()
    pyplot.savefig(save_path + '\\loss.png')
    pyplot.close()

    # save vit parameters
    pandas.DataFrame(vit_pars, index=['values'], columns=vit_pars.keys()).T \
        .to_csv(save_path + '\\vit_pars.csv', index=True)

    # save dino parameters
    pandas.DataFrame(dino_pars, index=['pars'], columns=dino_pars.keys()).T \
        .to_csv(save_path + '\\dino_pars.csv', index=True)

    print('history and parameters saved\n')


def run_training_for_64x64_cuts(epochs, grid=None, save_path='D:\ETH\projects\morpho-learner\\res\\dino\\'):

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'

    batch_size = 512
    N = 350000

    if grid is None:
        grid_size = 100
        grid = generate_grid(grid_size, 64, random_vit=True, random_dino=True)

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=lambda x: x / 255.)  # ~429000
    training_controls = CustomImageDataset(path_to_controls, 1, transform=lambda x: x / 255.)  # ~370000
    training_drugs, _ = torch.utils.data.random_split(training_drugs, [N, training_drugs.__len__() - N])
    training_controls, _ = torch.utils.data.random_split(training_controls, [N, training_controls.__len__() - N])

    joint_data = JointImageDataset([training_drugs, training_controls], transform=lambda x: x / 255., n_channels=3)
    data_loader = DataLoader(joint_data, batch_size=batch_size, shuffle=True)

    train(grid, epochs, data_loader, save_path)


def train(grid, epochs, data_loader, save_path):

    device = torch.device('cuda')
    for i, id in enumerate(grid['id']):

        print('training for set {} started'.format(id))
        if not os.path.exists(save_path + id):
            os.makedirs(save_path + id)

        model = ViT(**grid['vit'][i]).to(device)
        try:
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
                    print("epoch {}: {} min, loss = {:.4f}".format(epoch + 1, int((time.time() - start) / 60),
                                                                   epoch_loss))

                    # save network
                    torch.save(model.state_dict(), save_path + id + '\\ViT_at_{}.torch'.format(epoch))

                    if epoch >= 2:
                        if epoch_loss > loss_history[epoch - 1] > loss_history[epoch - 2] or epoch_loss > loss_history[
                            0]:
                            # if loss grows, stop training
                            break
                        elif round(epoch_loss, 4) == round(loss_history[epoch - 1], 4):
                            # if loss doesn't fall, stop
                            break

                print('{}/{} completed'.format(i + 1, len(grid['id'])))
                save_history_and_parameters(loss_history, grid['vit'][i], grid['dino'][i], save_path + id)

            except Exception as e:
                print('{}/{} failed with {}\n'.format(i + 1, len(grid['id']), e))
                shutil.rmtree(save_path + id)
        except Exception as e:
            print('{}/{} failed building dino with {}\n'.format(i + 1, len(grid['id']), e))
            shutil.rmtree(save_path + id)


def run_training_for_256x256_crops():
    """ Always out of memory on NVidia RTX 2060 :( """

    path_to_data = 'D:\ETH\projects\morpho-learner\data\cropped\max_conc\\'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\dino\\'

    grid_size = 100
    batch_size = 256
    N = 270450  # all images from max concentrations
    epochs = 10

    grid = generate_grid(grid_size, 256, random_vit=True, random_dino=True)  # run default parameters

    training_data = CustomImageDataset(path_to_data, 0, transform=lambda x: x / 255.)
    training_data, _ = torch.utils.data.random_split(training_data, [N, training_data.__len__() - N])
    three_channels = JointImageDataset([training_data], transform=lambda x: x / 255.)
    data_loader = DataLoader(three_channels, batch_size=batch_size, shuffle=True)

    train(grid, epochs, data_loader, save_path)


def get_image_tensor(path):
    image = numpy.array(Image.open(path).convert('RGB'))  # 3 channels
    image = numpy.moveaxis(image, -1, 0)  # set channels as the first dim
    image = torch.Tensor(image)  # make tensor
    image = image / 255.  # transform
    image = torch.unsqueeze(image, 0)  # add batch dim

    return image


def plot_attentions(attns):
    """ This is hardcoded for a particular architecture, so no general use. """
    fig, axs = pyplot.subplots(7, 3)
    for i in range(7):
        for j in range(3):
            axs[i, j].imshow(attns.numpy()[0][j][i])
            axs[i, j].set_title('layer {}, head {}'.format(j, i))
    pyplot.show()


def cluster_images(model_path, data_path):

    device = torch.device('cuda')
    vit_pars = pandas.read_csv(model_path + 'vit_pars.csv', index_col=0)
    # convert default types to int, except dropout ratios
    vit_pars = vit_pars.T.convert_dtypes(vit_par_types).T
    vit_pars = dict(vit_pars['values'])

    model = ViT(**vit_pars).to(device)
    checkpoint = [file for file in os.listdir(model_path) if file.endswith('.torch') and file.startswith('best')][0]
    model.load_state_dict(torch.load(model_path + checkpoint, map_location=device))
    model.eval()

    model = Recorder(model)  # to retrieve attentions and predictions

    for file in tqdm(os.listdir(data_path)):
        if file.endswith('.jpg'):

            image = get_image_tensor(data_path + file)
            # forward pass now returns predictions and the attention maps
            preds, attns = model(image.to(device))

            # retrieve predicted class
            predicted_class = str(int(torch.nn.Softmax(dim=1)(preds).argmax()))
            predicted_class_folder = model_path + 'clustering\\{}\\'.format(predicted_class)
            if not os.path.exists(predicted_class_folder):
                os.makedirs(predicted_class_folder)
            # copy file to a class folder
            shutil.copyfile(data_path+file, predicted_class_folder+file)


def train_best_models():
    """ Best models are hardcoded. """

    grid_size = 10

    grid_13d4753e = {'id': [], 'vit': [], 'dino': []}
    for _ in range(grid_size):
        grid_13d4753e['id'].append(str(uuid.uuid4())[:8])
        grid_13d4753e['vit'].append(
            dict(image_size=64, patch_size=16, num_classes=100, dim=64,
                 depth=5, heads=10, mlp_dim=128, dropout=0, emb_dropout=0.2)
        )
        grid_13d4753e['dino'].append(
            dict(image_size=64, hidden_layer='to_latent', projection_hidden_size=64, projection_layers=4,
                 num_classes_K=4096, student_temp=0.7, teacher_temp=0.09, local_upper_crop_scale=0.3,
                 global_lower_crop_scale=0.5, moving_average_decay=0.9, center_moving_average_decay=0.99)
        )
    save_path = 'D:\ETH\projects\morpho-learner\\res\dino\\grid_13d4753e\\'
    run_training_for_64x64_cuts(50, grid=grid_13d4753e, save_path=save_path)

    grid_60d3c055 = {'id': [], 'vit': [], 'dino': []}
    for _ in range(grid_size):
        grid_60d3c055['id'].append(str(uuid.uuid4())[:8])
        grid_60d3c055['vit'].append(
            dict(image_size=64, patch_size=8, num_classes=21, dim=64,
                 depth=3, heads=7, mlp_dim=64, dropout=0, emb_dropout=0.4)
        )
        grid_60d3c055['dino'].append(
            dict(image_size=64, hidden_layer='to_latent', projection_hidden_size=64, projection_layers=1,
                 num_classes_K=4096, student_temp=0.7, teacher_temp=0.05, local_upper_crop_scale=0.4,
                 global_lower_crop_scale=0.6, moving_average_decay=0.99, center_moving_average_decay=0.99)
        )
    save_path = 'D:\ETH\projects\morpho-learner\\res\dino\\grid_60d3c055\\'
    run_training_for_64x64_cuts(50, grid=grid_60d3c055, save_path=save_path)

    grid_bf723384 = {'id': [], 'vit': [], 'dino': []}
    for _ in range(grid_size):
        grid_bf723384['id'].append(str(uuid.uuid4())[:8])
        grid_bf723384['vit'].append(
            dict(image_size=64, patch_size=8, num_classes=1000, dim=64,
                 depth=2, heads=1, mlp_dim=256, dropout=0.2, emb_dropout=0.3)
        )
        grid_bf723384['dino'].append(
            dict(image_size=64, hidden_layer='to_latent', projection_hidden_size=64, projection_layers=2,
                 num_classes_K=8192, student_temp=0.8, teacher_temp=0.07, local_upper_crop_scale=0.5,
                 global_lower_crop_scale=0.5, moving_average_decay=0.99, center_moving_average_decay=0.99)
        )
    save_path = 'D:\ETH\projects\morpho-learner\\res\dino\\grid_bf723384\\'
    run_training_for_64x64_cuts(50, grid=grid_bf723384, save_path=save_path)


if __name__ == "__main__":

    run_training_for_64x64_cuts(10)