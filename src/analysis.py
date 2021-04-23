
import os, pandas, time, torch, numpy, umap, seaborn
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from torchvision.io import read_image
from src.models import Autoencoder


def plot_full_data_umap(data, parameters, labels, save_to):

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    for couple in parameters:
        neighbors, metric = couple

        reducer = umap.UMAP(n_neighbors=neighbors, metric=metric)
        start = time.time()
        embedding = reducer.fit_transform(scaled_data)
        print('umap transform with n = {} took {} s'.format(neighbors, time.time() - start))

        for hue in labels:
            pyplot.figure()
            seaborn.set(font_scale=0.5)
            seaborn.color_palette('colorblind')
            seaborn.axes_style('whitegrid')
            seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=hue, s=10)
            pyplot.title('UMAP: n={}, metric={}'.format(neighbors, metric), fontsize=8)
            pyplot.savefig(save_to + 'umap_hue={}_n={}_metric={}.png'.format(hue, neighbors, metric))
            pyplot.close('all')
            print('plot saved')


if __name__ == "__main__":

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\ae.torch'
    save_path = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\clustering\\'

    device = torch.device('cuda')

    # load trained autoencoder to use it in the transform
    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(path_to_ae_model, map_location=device))
    ae.eval()

    image_paths = [f for f in os.listdir(path_to_drugs)]
    cell_lines = [[f.split('_')[0] for f in os.listdir(path_to_drugs)]]
    plates = [[f.split('_')[2] for f in os.listdir(path_to_drugs)]]
    wells = [[f.split('_')[3] for f in os.listdir(path_to_drugs)]]
    drugs = [[f.split('_')[4] for f in os.listdir(path_to_drugs)]]
    dates = [['_'.join(f.split('_')[-3:]) for f in os.listdir(path_to_drugs)]]

    transform = lambda x: ae.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

    encodings = []
    for i, path in enumerate(image_paths):
        img = read_image(path_to_drugs + path)
        img_encoded = transform(img)

    encodings = numpy.array(encodings)

    umap_labels = [cell_lines, drugs]
    umap_pars = [(10, 'euclidean'), (25, 'euclidean'), (50, 'euclidean'), (100, 'euclidean'), (500, 'euclidean'),
                 (10, 'cosine'), (25, 'cosine'), (50, 'cosine'), (100, 'cosine'), (500, 'cosine'),
                 (10, 'correlation'), (25, 'correlation'), (50, 'correlation'), (100, 'correlation'), (500, 'correlation')]
    plot_full_data_umap(encodings, umap_pars, umap_labels, save_path)