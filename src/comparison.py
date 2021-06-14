import pandas, os, seaborn
from matplotlib import pyplot

from src.constants import cell_lines, drugs


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


if __name__ == "__main__":

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'

    plot_number_of_clusters('drugs', 300, save_to, filter_threshold=4)
    plot_number_of_clusters('cell_lines', 300, save_to, filter_threshold=4)

    plot_number_of_clusters('drugs', 30, save_to, filter_threshold=80)
    plot_number_of_clusters('cell_lines', 30, save_to, filter_threshold=80)
