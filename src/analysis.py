
import os, pandas, time, torch, numpy, umap, seaborn, shutil
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from torchvision.io import read_image
from tqdm import tqdm
from hdbscan import HDBSCAN
from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp, mannwhitneyu, kruskal, fisher_exact

from src.models import Autoencoder
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


def plot_drugs_umaps(path_to_drugs, save_path):

    for drug in tqdm(all_drugs):

        image_paths = [f for f in os.listdir(path_to_drugs) if drug in f]
        cell_lines = [f.split('_')[0] for f in image_paths]
        plates = [f.split('_')[2] for f in image_paths]
        wells = [f.split('_')[3] for f in image_paths]
        dates = ['_'.join(f.split('_')[-3:]) for f in image_paths]

        encodings = []
        for path in tqdm(image_paths):
            img = read_image(path_to_drugs + path)
            img_encoded = transform(img)
            encodings.append(img_encoded.detach().cpu().numpy())
        encodings = numpy.array(encodings)

        umap_labels = {'cell_line': cell_lines}
        umap_pars = [(10, 'euclidean'), (25, 'euclidean'), (50, 'euclidean'), (100, 'euclidean'), (500, 'euclidean'),
                     (10, 'cosine'), (25, 'cosine'), (50, 'cosine'), (100, 'cosine'), (500, 'cosine'),
                     (10, 'correlation'), (25, 'correlation'), (50, 'correlation'), (100, 'correlation'), (500, 'correlation')]

        plot_umap_with_pars_and_labels(encodings, umap_pars, umap_labels, save_path, umap_title=drug)


def plot_cluster_capacity(labels, grouping_factor, save_to):

    if not os.path.exists(save_to + '{}'.format(grouping_factor)):
        os.makedirs(save_to + '{}'.format(grouping_factor))

    clusters = sorted(list(set(labels)))
    cluster_capacity = pandas.DataFrame({'cluster': clusters, 'capacity': [0 for x in clusters]})
    for cluster in clusters:
        cluster_capacity.loc[cluster_capacity['cluster'] == cluster, 'capacity'] += sum([x == cluster for x in labels])

    pyplot.figure()
    seaborn.barplot(x='cluster', y='capacity', data=cluster_capacity)
    pyplot.xlabel('Cluster')
    pyplot.ylabel('Number of images')
    pyplot.title('Cluster capacity for {}'.format(grouping_factor))
    pyplot.grid()
    # pyplot.show()
    pyplot.savefig(save_to + '{}\\cluster_capacity_{}.pdf'.format(grouping_factor, grouping_factor))
    pyplot.close('all')
    print('cluster capacity plot saved')


def plot_cluster_enrichments_for_drug(clusters, cell_lines, drug, save_to):

    if not os.path.exists(save_to + drug):
        os.makedirs(save_to + drug)

    # compute cluster counts for cell lines
    unique_cls = list(set(cell_lines))
    unique_clusters = sorted(list(set(clusters)))
    cluster_counts = pandas.DataFrame(0, index=unique_clusters, columns=unique_cls)
    for i, cluster in enumerate(clusters):
        cluster_counts.loc[cluster, cell_lines[i]] += 1

    # calculate enrichments
    cluster_enrichments = calculate_cluster_enrichments(cluster_counts)

    # plot and save enrichments
    pyplot.figure(figsize=(12, 6))
    res = seaborn.heatmap(cluster_enrichments, xticklabels=cluster_enrichments.columns,
                          yticklabels=cluster_enrichments.index,
                          cmap='rocket', vmin=0, vmax=5)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=8)
    pyplot.title("Cluster enrichments for drug {}".format(drug))
    pyplot.tight_layout()
    pyplot.xticks(rotation=45)
    # pyplot.show()
    pyplot.savefig(save_to + drug + '\\cluster_enrichments_{}.pdf'.format(drug))
    pyplot.close('all')
    print("cluster enrichments plots for drug {} saved".format(drug))


def calculate_cluster_enrichments(cluster_counts):

    # calculate enrichments with cluster counts
    cluster_enrichments = cluster_counts[:]

    for i in range(cluster_counts.shape[0]):
        for j in range(cluster_counts.shape[1]):
            in_group_in_cluster = cluster_counts.iloc[i, j]
            in_group_not_in_cluster = cluster_counts.iloc[i, :].sum() - in_group_in_cluster
            not_in_group_in_cluster = cluster_counts.iloc[:, j].sum() - in_group_in_cluster
            not_in_group_not_in_cluster = cluster_counts.drop(cluster_counts.index[i]).drop(
                cluster_counts.columns[j], axis=1).sum().sum()

            _, p_value = fisher_exact([[in_group_in_cluster, in_group_not_in_cluster],
                                       [not_in_group_in_cluster, not_in_group_not_in_cluster]])

            cluster_enrichments.iloc[i, j] = p_value

    # correct for multiple hypothesis
    adj_p_values = multipletests(cluster_enrichments.values.flatten(), method='bonferroni')[1]
    for i in range(len(all_drugs)-1):
        adj_p_values = multipletests(adj_p_values, method='bonferroni')[1]

    # update the dataframe with adjusted values
    for i in range(cluster_enrichments.shape[0]):
        for j in range(cluster_enrichments.shape[1]):
            cluster_enrichments.iloc[i, j] = -numpy.log10(adj_p_values[i * cluster_enrichments.shape[1] + j])

    return cluster_enrichments


def save_cluster_members(path_to_all_images, drug_image_files, drug_image_clusters, drug, save_to, n=50):

    unique_clusters = sorted(list(set(drug_image_clusters)))

    for cluster in unique_clusters:
        if not os.path.exists(save_to + drug + '\\{}'.format(cluster)):
            os.makedirs(save_to + drug + '\\{}'.format(cluster))

        images_to_save = [f for f in drug_image_files if drug_image_clusters[drug_image_files.index(f)] == cluster][:n]
        # copy n representatives of cluster
        for image in images_to_save:
            shutil.copyfile(path_to_all_images + image, save_to + drug + "\\{}\\".format(cluster) + image)
    print("cluster members saved")


def plot_drugs_clustering(path_to_drugs, save_path):

    for drug in tqdm(all_drugs):

        image_paths = [f for f in os.listdir(path_to_drugs) if drug in f]
        cell_lines = [f.split('_')[0] for f in image_paths]
        plates = [f.split('_')[2] for f in image_paths]
        wells = [f.split('_')[3] for f in image_paths]
        dates = ['_'.join(f.split('_')[-3:]) for f in image_paths]

        # get encodings
        encodings = []
        for path in image_paths:
            img = read_image(path_to_drugs + path)
            img_encoded = transform(img)
            encodings.append(img_encoded.detach().cpu().numpy())
        encodings = numpy.array(encodings)

        # cluster encodings
        min_cluster_size = 20  # gives small noise percent & reasonable number of clusters
        min_samples = 1  # allow less conservative clustering

        normalized = normalize(encodings, norm='l2')  # after this 'euclidean' should approx. work as 'cosine'
        clusterer = HDBSCAN(metric='euclidean', min_samples=min_samples, min_cluster_size=min_cluster_size, allow_single_cluster=False)
        clusterer.fit(normalized)
        clusters = clusterer.labels_

        n_clusters = numpy.max(clusters) + 1
        noise = int(numpy.sum(clusters == -1) / len(clusters) * 100)
        print('clustering of drug {}:\n'.format(drug))
        print('min_cluster_size={}, min_samples={}, n clusters={}'.format(min_cluster_size, min_samples, n_clusters))
        print('noise={}%\n'.format(noise))

        # calculate and plot enrichments
        plot_cluster_enrichments_for_drug(clusters, cell_lines, drug, save_path)
        # plot cluster sizes
        plot_cluster_capacity(clusters, drug, save_path)
        # save cluster representatives
        save_cluster_members(path_to_drugs, image_paths, clusters, drug, save_path)


if __name__ == "__main__":

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\aecl_0.6672_0.8192_e100\\'

    device = torch.device('cuda')

    # load trained autoencoder to use it in the transform
    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(path_to_ae_model + 'ae.torch', map_location=device))
    ae.eval()

    transform = lambda x: ae.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

    # save_path = path_to_ae_model + 'full_data_umaps\\'
    # plot_full_data_umaps(path_to_drugs, save_path)

    # save_path = path_to_ae_model + 'cell_lines_umaps\\'
    # plot_cell_lines_umaps(path_to_drugs, save_path)

    # save_path = path_to_ae_model + 'drugs_umaps\\'
    # plot_drugs_umaps(path_to_drugs, save_path)

    save_path = path_to_ae_model + 'drugs_clustering\\'
    plot_drugs_clustering(path_to_drugs, save_path)
