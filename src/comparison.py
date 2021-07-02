import pandas, os, seaborn, numpy, umap, time
import torch
from matplotlib import pyplot
from tqdm import tqdm
from hdbscan import HDBSCAN

from src.constants import cell_lines, drugs
from src import analysis


def plot_number_of_clusters(key, mcs, save_to, filter_threshold=4):
    """
    :param key: 'drugs' or 'cell_lines'
    :param mcs: min_cluster_size used to cluster
    :param filter_threshold: filter out cell lines / drugs that had less clusters than value
    """

    if key == 'drugs':
        all_names = drugs
    elif key == 'cell_lines':
        all_names = cell_lines
    else:
        raise ValueError("Wrong key!")

    model_paths = {
        'unsupervised': 'D:\ETH\projects\morpho-learner\\res\\ae_at_100_0.667\\{}_clustering_mcs={}\\'.format(key, mcs),
        'weakly-supervised': 'D:\ETH\projects\morpho-learner\\res\dcl_at_100_0.7424\\{}_clustering_mcs={}\\'.format(key, mcs),
        'adversarial': 'D:\ETH\projects\morpho-learner\\res\\aecl_at_100_0.667_0.7743\\{}_clustering_mcs={}\\'.format(key, mcs),
        'self-supervised': 'D:\ETH\projects\morpho-learner\\res\dcl+byol_at_17\\{}_clustering_mcs={}\\'.format(key, mcs)
    }

    names = []
    methods = []
    n_clusters = []
    for method in model_paths.keys():
        for i, name in enumerate(all_names):
            if name != 'DMSO':
                # calculate number of clusters for a given method and a given cell line or drug
                n = len([f for f in os.listdir(model_paths[method] + name) if os.path.isdir(model_paths[method] + name + '\\' + f)])

                n_clusters.append(n)
                names.append(name)
                methods.append(method)

    results = pandas.DataFrame({'method': methods, 'name': names, 'N': n_clusters})
    results = results.sort_values('N')

    if filter_threshold > 0:
        for name in all_names:
            name_sum = results.loc[results['name'] == name, 'N'].sum()
            if name_sum < filter_threshold * len(model_paths):
                # if all methods have less than 4 clusters, filter out this name
                results = results.drop(results[results['name'] == name].index)

    plot_title = "Clustering of {}".format(key.replace('_', ' '))
    pyplot.figure(figsize=(12,6))
    seaborn.set_theme(style="whitegrid")
    seaborn.barplot(x='name', y='N', hue='method', data=results)
    pyplot.title(plot_title)
    pyplot.xticks(rotation=45)
    pyplot.legend(bbox_to_anchor=(1.01, 1))
    pyplot.tight_layout()
    pyplot.savefig(save_to + plot_title.replace(' ', '_') + '_mcs={}.pdf'.format(mcs))


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_drugs, save_to):

    results = {'drug': [], 'model': [], 'min_cluster_size': [], 'n_clusters': [], 'noise': []}

    for drug in tqdm(drugs):
        for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:

            transform = analysis.get_f_transform(model, torch.device('cuda'))
            encodings, image_ids = analysis.get_image_encodings_from_path(path_to_drugs, drug, transform)
            encodings = numpy.array(encodings)

            for min_cluster_size in tqdm(range(10, 310, 10)):

                start = time.time()
                reducer = umap.UMAP(n_neighbors=min_cluster_size, metric='euclidean')
                embedding = reducer.fit_transform(encodings)
                # cluster encodings
                clusterer = HDBSCAN(metric='euclidean', min_samples=1, min_cluster_size=min_cluster_size, allow_single_cluster=False)
                clusterer.fit(embedding)
                clusters = clusterer.labels_
                print('umap + hdbscan clustering with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

                n_clusters = numpy.max(clusters) + 1
                noise = int(numpy.sum(clusters == -1) / len(clusters) * 100)

                results['drug'].append(drug)
                results['model'].append(model)
                results['min_cluster_size'].append(min_cluster_size)
                results['n_clusters'].append(n_clusters)
                results['noise'].append(noise)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'clustering_over_parameter_sets.csv', index=False)


if __name__ == "__main__":

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'

    # plot_number_of_clusters('drugs', 300, save_to, filter_threshold=4)
    # plot_number_of_clusters('cell_lines', 300, save_to, filter_threshold=4)
    # plot_number_of_clusters('drugs', 30, save_to, filter_threshold=80)
    # plot_number_of_clusters('cell_lines', 30, save_to, filter_threshold=80)

    path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'

    collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_drugs, save_to)