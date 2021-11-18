
import os, pandas, time, torch, numpy, uuid, seaborn, random, shutil, traceback, copy
from PIL import Image
from matplotlib import pyplot
from functools import wraps
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T

from src.models import DeepClassifier
from src.datasets import CustomImageDataset, JointImageDataset


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):
    # BYOL loss
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class RandomApply(torch.nn.Module):
    # for random blur
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class EMA():
    # exponential moving average
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MLP(torch.nn.Module):
    # MLP class for projector and predictor
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(torch.nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOL(torch.nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        augment_fn2=None,
        moving_average_decay=0.99,
        use_momentum=True
    ):
        super().__init__()
        self.net = net

        # default SimCLR does not apply to our data, so
        DEFAULT_AUG = torch.nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            RandomApply(T.GaussianBlur((3, 3), (.1, 2.0)), p=0.2),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(mean=torch.tensor([0.449]), std=torch.tensor([0.226]))
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # pass mock 1-channel tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 1, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


def get_byol_pars(im_size, transform=None, randomize=True):

    if randomize:
        return dict(image_size=im_size,  # 256
                    hidden_layer='model.9',
                    projection_size=random.sample([64, 128, 256, 512, 1024, 2048], 1)[0],  # 256
                    projection_hidden_size=random.sample([512, 1024, 2048, 4096, 8192], 1)[0],  # 4096
                    augment_fn=transform,
                    augment_fn2=transform,
                    moving_average_decay=random.sample([0.8, 0.9, 0.99], 1)[0],
                    use_momentum=True)
    else:
        return dict(image_size=im_size,
                    hidden_layer='model.9',
                    projection_size=64,
                    projection_hidden_size=2048,
                    augment_fn=transform,
                    augment_fn2=transform,
                    moving_average_decay=0.8,
                    use_momentum=True)


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


def run_training_for_64x64_cuts(epochs, data_loader, device=torch.device('cuda'), grid=None, run_id=""):

    save_path = 'D:\ETH\projects\morpho-learner\\res\\byol\\{}\\'.format(run_id)

    model = DeepClassifier().to(device)

    if grid is None:
        grid_size = 1
        grid = generate_grid(grid_size, 64, randomize=False)

    train(model, grid, epochs, data_loader, device, save_path)


def train(model, grid, epochs, data_loader, device, save_path):

    for i, id in enumerate(grid['id']):

        print(pandas.DataFrame(grid['byol'][i], index=['values'], columns=grid['byol'][i].keys()).T.to_string())
        print('training for set {} started'.format(id))
        if not os.path.exists(save_path + id):
            os.makedirs(save_path + id)

        try:
            learner = BYOL(model, **grid['byol'][i]).to(device)
            optimizer = torch.optim.Adam(learner.parameters(), lr=0.0001)
            scheduler = None

            loss_history = []
            try:
                for epoch in range(epochs):
                    start = time.time()
                    epoch_loss = 0
                    n_crops = 1
                    for batch in data_loader:
                        n_crops = len(batch)
                        for crops, _ in batch:
                            crops = crops.float().to(device)
                            loss = learner(crops)
                            epoch_loss += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            # update moving average of teacher encoder and teacher centers
                            learner.update_moving_average()

                    epoch_loss = epoch_loss / len(data_loader) / n_crops
                    loss_history.append(epoch_loss)
                    print("epoch {}: {} min, loss = {:.4f}".format(epoch + 1, int((time.time() - start) / 60), epoch_loss))

                    # update lr
                    if scheduler is not None:
                        scheduler.step()

                    # save network
                    torch.save(model.state_dict(), save_path + id + '\\byol_at_{}.torch'.format(epoch+1))

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
                # shutil.rmtree(save_path + id)
        except Exception as e:
            print('{}/{} failed building byol with {}\n'.format(i + 1, len(grid['id']), e))
            print(traceback.print_exc())
            # shutil.rmtree(save_path + id)


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