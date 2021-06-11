
import os, pandas, time, torch, numpy, umap, seaborn, shutil, random
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from torchvision.io import read_image
from tqdm import tqdm
from hdbscan import HDBSCAN
from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp, mannwhitneyu, kruskal, fisher_exact
from torch.nn import Sequential

from src.models import Autoencoder, DeepClassifier
from src.constants import cell_lines as all_cell_lines
from src.constants import drugs as all_drugs
from src.constants import get_type_by_name


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


def plot_full_data_umaps(path_to_drugs, save_path, transform):

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
    umap_pars = [(300, 'euclidean'), (600, 'euclidean'), (1200, 'euclidean')]

    plot_umap_with_pars_and_labels(encodings, umap_pars, umap_labels, save_path, umap_title='full')


def plot_cell_lines_umaps(path_to_drugs, save_path, transform):

    for cell_line in tqdm(all_cell_lines):

        image_paths = [f for f in os.listdir(path_to_drugs) if cell_line in f]
        plates = [f.split('_')[2] for f in image_paths]
        wells = [f.split('_')[3] for f in image_paths]
        drugs = [f.split('_')[4] for f in image_paths]
        dates = ['_'.join(f.split('_')[-3:]) for f in image_paths]

        encodings = []
        for path in image_paths:
            img = read_image(path_to_drugs + path)
            img_encoded = transform(img)
            encodings.append(img_encoded.detach().cpu().numpy())
        encodings = numpy.array(encodings)

        umap_labels = {'drug': drugs}
        umap_pars = [(25, 'euclidean'), (25, 'cosine'), (25, 'correlation')]

        plot_umap_with_pars_and_labels(encodings, umap_pars, umap_labels, save_path, umap_title=cell_line)


def plot_drugs_umaps(path_to_drugs, save_path, transform):

    for drug in tqdm(all_drugs):

        image_paths = [f for f in os.listdir(path_to_drugs) if drug in f]
        cell_lines = [f.split('_')[0] for f in image_paths]
        plates = [f.split('_')[2] for f in image_paths]
        wells = [f.split('_')[3] for f in image_paths]
        dates = ['_'.join(f.split('_')[-3:]) for f in image_paths]

        encodings = []
        for path in image_paths:
            img = read_image(path_to_drugs + path)
            img_encoded = transform(img)
            encodings.append(img_encoded.detach().cpu().numpy())
        encodings = numpy.array(encodings)

        umap_labels = {'cell_line': cell_lines}
        umap_pars = [(25, 'euclidean'), (25, 'cosine'), (25, 'correlation')]

        plot_umap_with_pars_and_labels(encodings, umap_pars, umap_labels, save_path, umap_title=drug)


def plot_cluster_capacity(labels, grouping_factor, save_to):

    if not os.path.exists(save_to + '{}'.format(grouping_factor)):
        os.makedirs(save_to + '{}'.format(grouping_factor))

    clusters = sorted(list(set(labels)))
    cluster_capacity = pandas.DataFrame({'cluster': clusters, 'capacity': [0 for x in clusters]})
    for cluster in clusters:
        cluster_capacity.loc[cluster_capacity['cluster'] == cluster, 'capacity'] += sum([x == cluster for x in labels])

    pyplot.figure(figsize=(12,6))
    seaborn.barplot(x='cluster', y='capacity', data=cluster_capacity)
    pyplot.xlabel('Cluster')
    pyplot.ylabel('Number of images')
    pyplot.title('Cluster capacity for {}'.format(grouping_factor))
    pyplot.grid()
    # pyplot.show()
    pyplot.savefig(save_to + '{}\\cluster_capacity_{}.pdf'.format(grouping_factor, grouping_factor))
    pyplot.close('all')
    print('cluster capacity plot saved')


def plot_cluster_enrichments(clusters, image_ids, image_grouping_factor, save_to):

    if not os.path.exists(save_to + image_grouping_factor):
        os.makedirs(save_to + image_grouping_factor)

    # compute cluster counts for cell lines
    unique_cls = list(set(image_ids))
    unique_clusters = sorted(list(set(clusters)))
    cluster_counts = pandas.DataFrame(0, index=unique_clusters, columns=unique_cls)
    for i, cluster in enumerate(clusters):
        cluster_counts.loc[cluster, image_ids[i]] += 1

    # calculate enrichments
    cluster_enrichments = calculate_cluster_enrichments(cluster_counts)

    # plot and save enrichments
    pyplot.figure(figsize=(12, 6))
    res = seaborn.heatmap(cluster_enrichments, xticklabels=cluster_enrichments.columns,
                          yticklabels=cluster_enrichments.index,
                          cmap='rocket', vmin=0, vmax=5)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=8)
    pyplot.title("Cluster enrichments for {} {}".format(get_type_by_name(image_grouping_factor), image_grouping_factor))
    pyplot.tight_layout()
    pyplot.xticks(rotation=45)
    # pyplot.show()
    pyplot.savefig(save_to + image_grouping_factor + '\\cluster_enrichments_{}.pdf'.format(image_grouping_factor))
    pyplot.close('all')
    print("cluster enrichments plots for {} {} saved".format(get_type_by_name(image_grouping_factor), image_grouping_factor))


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
    for i in range(len(all_drugs)-2):
        adj_p_values = multipletests(adj_p_values, method='bonferroni')[1]

    # update the dataframe with adjusted values
    for i in range(cluster_enrichments.shape[0]):
        for j in range(cluster_enrichments.shape[1]):
            cluster_enrichments.iloc[i, j] = -numpy.log10(adj_p_values[i * cluster_enrichments.shape[1] + j])

    return cluster_enrichments


def save_cluster_members(path_to_drug_images, image_files, image_clusters, image_grouping_factor, save_to, n=None,  path_to_control_images=None):

    unique_clusters = sorted(list(set(image_clusters)))

    for cluster in unique_clusters:
        if not os.path.exists(save_to + image_grouping_factor + '\\{}'.format(cluster)):
            os.makedirs(save_to + image_grouping_factor + '\\{}'.format(cluster))

        images_to_save = [f for f in image_files if image_clusters[image_files.index(f)] == cluster]
        if n is not None:
            images_to_save = random.sample(images_to_save, min(n, len(images_to_save)))

        # copy n representatives of cluster
        for image in images_to_save:
            if path_to_control_images is None:
                shutil.copyfile(path_to_drug_images + image, save_to + image_grouping_factor + "\\{}\\".format(cluster) + image)
            else:
                if 'DMSO' in image:
                    shutil.copyfile(path_to_control_images + image, save_to + image_grouping_factor + "\\{}\\".format(cluster) + image)
                else:
                    shutil.copyfile(path_to_drug_images + image, save_to + image_grouping_factor + "\\{}\\".format(cluster) + image)

    print("cluster members saved")


def plot_drugs_clustering(min_cluster_size, path_to_drugs, save_path, transform):

    for drug in tqdm(all_drugs):

        if not os.path.exists(save_path + drug):
            os.makedirs(save_path + drug)

        encodings, image_ids = get_image_encodings_from_path(path_to_drugs, drug, transform)
        encodings = numpy.array(encodings)

        reducer = umap.UMAP(n_neighbors=min_cluster_size, metric='euclidean')
        start = time.time()
        embedding = reducer.fit_transform(encodings)
        print('umap transform with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

        # cluster encodings
        start = time.time()
        clusterer = HDBSCAN(metric='euclidean', min_samples=1, min_cluster_size=min_cluster_size, allow_single_cluster=False)
        clusterer.fit(embedding)
        clusters = clusterer.labels_
        print('hdbscan clustering with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

        n_clusters = numpy.max(clusters) + 1
        noise = int(numpy.sum(clusters == -1) / len(clusters) * 100)
        print('min_cluster_size={}, min_samples=1, n clusters={}'.format(min_cluster_size, n_clusters))
        print('noise={}%\n'.format(noise))

        # filter out -1 ('noise') cluster
        x, y = [], []
        cells_labels, clusters_labels = [], []
        filenames = []
        for i in range(embedding.shape[0]):
            if clusters[i] != -1:
                x.append(embedding[i,0])
                y.append(embedding[i,1])
                cells_labels.append(image_ids['cell_lines'][i])
                clusters_labels.append(str(clusters[i]))
                filenames.append(image_ids['filenames'][i])

        # umap coloring cell lines
        pyplot.figure()
        seaborn.set(font_scale=0.5)
        seaborn.color_palette('colorblind')
        seaborn.axes_style('whitegrid')
        seaborn.scatterplot(x=x, y=y, hue=cells_labels, s=10)
        pyplot.legend()
        pyplot.title('{} UMAP: n={}, metric=euclidean'.format(drug, min_cluster_size), fontsize=8)
        pyplot.savefig(save_path + '{}\\{}_umap_hue=cell_line_n={}_metric=euclidean.pdf'.format(drug, drug, min_cluster_size), dpi=300)
        pyplot.close('all')

        if n_clusters > 15:
            print("too many clusters to plot\n")
        else:
            # umap coloring clusters
            pyplot.figure()
            seaborn.set(font_scale=0.5)
            seaborn.color_palette('colorblind')
            seaborn.axes_style('whitegrid')
            seaborn.scatterplot(x=x, y=y, hue=clusters_labels, s=10)
            pyplot.legend()
            pyplot.title('{} UMAP: n={}, metric=euclidean'.format(drug, min_cluster_size), fontsize=8)
            pyplot.savefig(save_path + '{}\\{}_umap_hue=clusters_n={}_metric=euclidean.pdf'.format(drug, drug, min_cluster_size), dpi=300)
            pyplot.close('all')

        # calculate and plot enrichments
        plot_cluster_enrichments(clusters_labels, cells_labels, drug, save_path)
        # plot cluster sizes
        plot_cluster_capacity(clusters_labels, drug, save_path)
        # save cluster representatives
        save_cluster_members(path_to_drugs, filenames, clusters_labels, drug, save_path)


def get_image_encodings_from_path(path, common_image_id, transform, n=None):

    encodings = []

    filenames = [f for f in os.listdir(path) if common_image_id in f]
    if n is not None:
        filenames = random.sample(filenames, n)

    image_ids = {
        'filenames': filenames,
        'cell_lines': [f.split('_')[0] for f in filenames],
        'plates': [f.split('_')[2] for f in filenames],
        'wells': [f.split('_')[3] for f in filenames],
        'drugs': [f.split('_')[4] for f in filenames],
        'dates': ['_'.join(f.split('_')[-3:]) for f in filenames]
    }

    # get encodings
    for file in filenames:
        img = read_image(path + file)
        img_encoded = transform(img)
        encodings.append(img_encoded.detach().cpu().numpy())

    return encodings, image_ids


def plot_cell_lines_clustering(min_cluster_size, path_to_drugs, path_to_controls, save_path, transform):

    for cell_line in tqdm(all_cell_lines):

        if not os.path.exists(save_path + cell_line):
            os.makedirs(save_path + cell_line)

        drugs_encodings, drugs_ids = get_image_encodings_from_path(path_to_drugs, cell_line, transform)
        controls_encodings, controls_ids = get_image_encodings_from_path(path_to_controls, cell_line, transform, n=int(0.25 * len(drugs_encodings)))

        encodings = numpy.array([*drugs_encodings, *controls_encodings])
        drugs = [*drugs_ids['drugs'], *controls_ids['drugs']]
        image_filenames = [*drugs_ids['filenames'], *controls_ids['filenames']]

        reducer = umap.UMAP(n_neighbors=min_cluster_size, metric='euclidean')
        start = time.time()
        embedding = reducer.fit_transform(encodings)
        print('umap transform with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

        # cluster encodings
        start = time.time()
        clusterer = HDBSCAN(metric='euclidean', min_samples=1, min_cluster_size=min_cluster_size, allow_single_cluster=False)
        clusterer.fit(embedding)
        clusters = clusterer.labels_
        print('hdbscan clustering with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

        n_clusters = numpy.max(clusters) + 1
        noise = int(numpy.sum(clusters == -1) / len(clusters) * 100)
        print('clustering of cell line {}:\n'.format(cell_line))
        print('min_cluster_size={}, min_samples=1, n clusters={}'.format(min_cluster_size, n_clusters))
        print('noise={}%\n'.format(noise))

        # filter out -1 ('noise') cluster
        x, y = [], []
        drugs_labels, clusters_labels = [], []
        filenames = []
        for i in range(embedding.shape[0]):
            if clusters[i] != -1:
                x.append(embedding[i, 0])
                y.append(embedding[i, 1])
                drugs_labels.append(drugs[i])
                clusters_labels.append(str(clusters[i]))
                filenames.append(image_filenames[i])

        # umap coloring drugs
        pyplot.figure()
        seaborn.set(font_scale=0.5)
        seaborn.color_palette('colorblind')
        seaborn.axes_style('whitegrid')
        seaborn.scatterplot(x=x, y=y, hue=drugs_labels, s=10)
        pyplot.legend()
        pyplot.title('{} UMAP: n={}, metric=euclidean'.format(cell_line, min_cluster_size), fontsize=8)
        pyplot.savefig(save_path + '{}\\{}_umap_hue=drug_n={}_metric=euclidean.pdf'.format(cell_line, cell_line, min_cluster_size), dpi=300)
        pyplot.close('all')

        if n_clusters > 15:
            print("too many clusters to plot\n")
        else:
            # umap coloring clusters
            pyplot.figure()
            seaborn.set(font_scale=0.5)
            seaborn.color_palette('colorblind')
            seaborn.axes_style('whitegrid')
            seaborn.scatterplot(x=x, y=y, hue=clusters_labels, s=10)
            pyplot.legend()
            pyplot.title('{} UMAP: n={}, metric=euclidean'.format(cell_line, min_cluster_size), fontsize=8)
            pyplot.savefig(save_path + '{}\\{}_umap_hue=clusters_n={}_metric=euclidean.pdf'.format(cell_line, cell_line, min_cluster_size), dpi=300)
            pyplot.close('all')

        # calculate and plot enrichments
        plot_cluster_enrichments(clusters_labels, drugs_labels, cell_line, save_path)
        # plot cluster sizes
        plot_cluster_capacity(clusters_labels, cell_line, save_path)
        # save cluster representatives
        save_cluster_members(path_to_drugs, filenames, clusters_labels, cell_line, save_path,
                             path_to_control_images=path_to_controls)


if __name__ == "__main__":

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
    path_to_controls = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'

    device = torch.device('cuda')

    analyze_unsupervised = True
    analyze_weakly_supervised = True
    analyze_adversarial = True
    analyze_self_supervised = True

    min_cluster_size = 300

    if analyze_unsupervised:
        path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\ae_at_100_0.667\\'
        model = Autoencoder().to(device)
        # load a trained autoencoder to use it in the transform
        model.load_state_dict(torch.load(path_to_ae_model + 'autoencoder.torch', map_location=device)).eval()
        # create a transform function with autoencoder
        transform = lambda x: model.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

        # run the analysis for unsupervised approach
        plot_cell_lines_clustering(min_cluster_size, path_to_drugs, path_to_controls, path_to_ae_model + 'new_cell_lines_clustering\\', transform)
        plot_drugs_clustering(min_cluster_size, path_to_drugs, path_to_ae_model + 'new_drugs_clustering\\', transform)
        plot_full_data_umaps(path_to_drugs, path_to_ae_model + 'full_data_umaps\\', transform)

    if analyze_adversarial:
        path_to_ae_model = 'D:\ETH\projects\morpho-learner\\res\\aecl_at_100_0.667_0.7743\\'
        model = Autoencoder().to(device)
        # load a trained autoencoder to use it in the transform
        model.load_state_dict(torch.load(path_to_ae_model + 'ae.torch', map_location=device)).eval()
        # create a transform function with autoencoder
        transform = lambda x: model.encoder(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

        # run the analysis for adversarial approach
        plot_cell_lines_clustering(path_to_drugs, path_to_controls, path_to_ae_model + 'new_cell_lines_clustering\\', transform)
        plot_drugs_clustering(path_to_drugs, path_to_ae_model + 'new_drugs_clustering\\', transform)
        plot_full_data_umaps(path_to_drugs, path_to_ae_model + 'full_data_umaps\\', transform)

    if analyze_weakly_supervised:
        path_to_cl_model = 'D:\ETH\projects\morpho-learner\\res\\dcl_at_100_0.7424\\'
        model = DeepClassifier().to(device)
        # load a trained deep classifier to use it in the transform
        model.load_state_dict(torch.load(path_to_cl_model + 'deep_classifier.torch', map_location=device)).eval()
        # truncate to the layer with learned representations
        model = Sequential(*list(model.model.children())[:-4])
        # create a transform function with weakly supervised classifier
        transform = lambda x: model(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

        # run the analysis for weakly supervised approach
        plot_cell_lines_clustering(min_cluster_size, path_to_drugs, path_to_controls, path_to_cl_model + 'new_cell_lines_clustering\\', transform)
        plot_drugs_clustering(min_cluster_size, path_to_drugs, path_to_cl_model + 'new_drugs_clustering\\', transform)
        plot_full_data_umaps(path_to_drugs, path_to_cl_model + 'full_data_umaps\\', transform)

    if analyze_self_supervised:
        path_to_cl_model = 'D:\ETH\projects\morpho-learner\\res\dcl+byol_at_17\\'
        model = DeepClassifier().to(device)
        # load a trained deep classifier to use it in the transform
        model.load_state_dict(torch.load(path_to_cl_model + 'best_dcl+byol_at_16.torch', map_location=device)).eval()
        # truncate to the layer with learned representations
        model = Sequential(*list(model.model.children())[:-4])
        # create a transform function with weakly supervised classifier
        transform = lambda x: model(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

        # run the analysis for weakly supervised approach
        plot_cell_lines_clustering(min_cluster_size, path_to_drugs, path_to_controls, path_to_cl_model + 'new_cell_lines_clustering\\', transform)
        plot_drugs_clustering(min_cluster_size, path_to_drugs, path_to_cl_model + 'new_drugs_clustering\\', transform)
        plot_full_data_umaps(path_to_drugs, path_to_cl_model + 'full_data_umaps\\', transform)
