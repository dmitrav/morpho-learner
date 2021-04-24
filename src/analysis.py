
import os, pandas, time, torch, numpy, umap, seaborn
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from torchvision.io import read_image
from src.models import Autoencoder
from tqdm import tqdm
from src.constants import cell_lines as all_cell_lines
from src.constants import drugs as all_drugs


def plot_umap_with_pars_and_labels(data, parameters, labels, save_to, umap_title=""):

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    scaled_data = StandardScaler().fit_transform(data)

    for couple in parameters:
        neighbors, metric = couple

        reducer = umap.UMAP(n_neighbors=neighbors, metric=metric)
        start = time.time()
        embedding = reducer.fit_transform(scaled_data)
        print('\numap transform with {} metric, n = {}, took {} s'.format(metric, neighbors, int(time.time() - start)))

        for hue in labels.keys():
            pyplot.figure()
            seaborn.set(font_scale=0.5)
            seaborn.color_palette('colorblind')
            seaborn.axes_style('whitegrid')
            seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels[hue], s=10)
            pyplot.title('{} UMAP: n={}, metric={}'.format(umap_title, neighbors, metric), fontsize=8)
            pyplot.savefig(save_to + '{}_umap_hue={}_n={}_metric={}.png'.format(umap_title, hue, neighbors, metric), dpi=300)
            pyplot.close('all')
            print('{}_umap_hue={}_n={}_metric={}.png saved'.format(umap_title, hue, neighbors, metric))


def plot_full_data_umaps(path_to_drugs, save_path):

    image_paths = [f for f in os.listdir(path_to_drugs)[::2]]
    cell_lines = [f.split('_')[0] for f in image_paths]
    plates = [f.split('_')[2] for f in image_paths]
    wells = [f.split('_')[3] for f in image_paths]
    drugs = [f.split('_')[4] for f in image_paths]
    dates = ['_'.join(f.split('_')[-3:]) for f in image_paths]

    encodings = []
    for path in tqdm(image_paths):
        img = read_image(path_to_drugs + path)
        img_encoded = transform(img)
        encodings.append(img_encoded.detach().cpu().numpy())
    encodings = numpy.array(encodings)

    umap_labels = {'cell_line': cell_lines, 'drug': drugs}
    umap_pars = [(10, 'euclidean'), (25, 'euclidean'), (50, 'euclidean'), (100, 'euclidean'), (500, 'euclidean'),
                 (10, 'cosine'), (25, 'cosine'), (50, 'cosine'), (100, 'cosine'), (500, 'cosine'),
                 (10, 'correlation'), (25, 'correlation'), (50, 'correlation'), (100, 'correlation'), (500, 'correlation')]

    plot_umap_with_pars_and_labels(encodings, umap_pars, umap_labels, save_path)


def plot_cell_lines_umaps(path_to_drugs, save_path):

    for cell_line in tqdm(all_cell_lines):

        image_paths = [f for f in os.listdir(path_to_drugs) if cell_line in f]
        cell_lines = [f.split('_')[0] for f in image_paths]
        plates = [f.split('_')[2] for f in image_paths]
        wells = [f.split('_')[3] for f in image_paths]
        drugs = [f.split('_')[4] for f in image_paths]
        dates = ['_'.join(f.split('_')[-3:]) for f in image_paths]

        encodings = []
        for path in tqdm(image_paths):
            img = read_image(path_to_drugs + path)
            img_encoded = transform(img)
            encodings.append(img_encoded.detach().cpu().numpy())
        encodings = numpy.array(encodings)

        umap_labels = {'drug': drugs}
        umap_pars = [(10, 'euclidean'), (25, 'euclidean'), (50, 'euclidean'), (100, 'euclidean'), (500, 'euclidean'),
                     (10, 'cosine'), (25, 'cosine'), (50, 'cosine'), (100, 'cosine'), (500, 'cosine'),
                     (10, 'correlation'), (25, 'correlation'), (50, 'correlation'), (100, 'correlation'), (500, 'correlation')]

        plot_umap_with_pars_and_labels(encodings, umap_pars, umap_labels, save_path, umap_title=cell_line)


if __name__ == "__main__":

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\ae.torch'

    device = torch.device('cuda')

    # load trained autoencoder to use it in the transform
    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(path_to_ae_model, map_location=device))
    ae.eval()

    transform = lambda x: ae.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

    # save_path = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\full_data_umaps\\'
    # plot_full_data_umaps(path_to_drugs, save_path)

    save_path = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\cell_lines_umaps\\'
    plot_cell_lines_umaps(path_to_drugs, save_path)
