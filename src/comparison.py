import pandas, os, seaborn, numpy, umap, time, torch, scipy
from sklearn import metrics
from matplotlib import pyplot
from torchmetrics import metric
from tqdm import tqdm
from hdbscan import HDBSCAN

from src.constants import cell_lines, drugs
from src import analysis


def plot_number_of_clusters(key, mcs, save_to, filter_threshold=4):
    """ Evaluate clustering made previously for certain parameters.
    :param key: 'drugs' or 'cell_lines'
    :param mcs: min_cluster_size used to cluster
    :param filter_threshold: filter out cell lines / drugs that had less clusters than value
    """

    if key == 'drugs':
        # all_names = drugs
        all_names = ['Clofarabine', 'Pemetrexed', 'Irinotecan', 'Docetaxel',
                     'Trametinib', 'Rapamycin', 'Paclitaxel', 'Methotrexate']

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

    results = pandas.DataFrame({'method': methods, 'name': names, 'Number of clusters': n_clusters})
    results = results.sort_values('Number of clusters')

    if filter_threshold > 0:
        for name in all_names:
            name_sum = results.loc[results['name'] == name, 'Number of clusters'].sum()
            if name_sum < filter_threshold * len(model_paths):
                # if all methods have less than 4 clusters, filter out this name
                results = results.drop(results[results['name'] == name].index)

    plot_title = "Clustering of {}".format(key.replace('_', ' '))
    # pyplot.figure(figsize=(12,6))
    # pyplot.figure(figsize=(10,6))
    seaborn.set_theme(style="whitegrid")
    seaborn.barplot(x='name', y='Number of clusters', hue='method', data=results)
    # pyplot.title(plot_title)
    pyplot.xticks(rotation=45)
    pyplot.ylim(0, 19)
    pyplot.yticks([x for x in range(0, 13, 2)])
    # pyplot.legend(bbox_to_anchor=(1.01, 1))
    pyplot.tight_layout()
    pyplot.savefig(save_to + plot_title.replace(' ', '_') + '_mcs={}.pdf'.format(mcs))


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_drugs, save_to):
    """ Cluster the dataset over multiple parameters, evaluate results and save results as a dataframe. """

    results = {'drug': [], 'model': [], 'min_cluster_size': [], 'n_clusters': [],
               'noise': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}

    for drug in tqdm(drugs):

        if drug in ['PBS', 'DMSO']:
            continue
        else:
            for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:

                transform = analysis.get_f_transform(model, torch.device('cuda'))
                encodings, image_ids = analysis.get_image_encodings_from_path(path_to_drugs, drug, transform)
                encodings = numpy.array(encodings)

                for min_cluster_size in tqdm(range(30, 310, 30)):

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
                    silhouette = metrics.silhouette_score(embedding, clusters)
                    calinski_harabasz = metrics.calinski_harabasz_score(embedding, clusters)
                    davies_bouldin = metrics.davies_bouldin_score(embedding, clusters)

                    results['drug'].append(drug)
                    results['model'].append(model)
                    results['min_cluster_size'].append(min_cluster_size)
                    results['n_clusters'].append(n_clusters)
                    results['noise'].append(noise)
                    results['silhouette'].append(silhouette)
                    results['calinski_harabasz'].append(calinski_harabasz)
                    results['davies_bouldin'].append(davies_bouldin)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'clusters_over_min_cluster_size.csv', index=False)


def plot_full_distribution_of_clusters(clustering_results, save_to):

    seaborn.set_theme(style="whitegrid")
    seaborn.displot(clustering_results, x='n_clusters', hue='method', multiple="dodge", binwidth=10)
    pyplot.xlabel('Number of clusters')
    pyplot.savefig(save_to + 'clusters_over_min_cluster_size.pdf')


def select_the_best_clustering_results(clustering_results):

    # find uncorrelated metrics for further evaluations
    print("corr(silhouette, calinski_harabasz) = {}\ncorr(silhouette, davies_bouldin) = {}".format(
        scipy.stats.pearsonr(clustering_results.loc[:, 'silhouette'], clustering_results.loc[:, 'calinski_harabasz'])[0],
        scipy.stats.pearsonr(clustering_results.loc[:, 'silhouette'], clustering_results.loc[:, 'davies_bouldin'])[0]))

    best_clustering = pandas.DataFrame(columns=clustering_results.columns)
    for method in methods:
        for drug in drugs:
            df = clustering_results.loc[(clustering_results['drug'] == drug) & (clustering_results['method'] == method), :]
            # select the best results by three metrics:
            df = df[df['silhouette'] >= df['silhouette'].median()]
            df = df[df['davies_bouldin'] <= df['davies_bouldin'].median()]
            df = df[df['noise'] == df['noise'].min()]
            if df.shape[0] > 1:
                # if there are multiple equivalent results, take more clusters
                df = df[df['n_clusters'] == df['n_clusters'].max()]
            best_clustering = pandas.concat([best_clustering, df])

    return best_clustering


def print_statistics_on_clustering_results(clustering_results):

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'adversarial']:
        print(
            "{}:\nmean (median) n_clusters = {} ({})\nmedian noise = {}%\nmax silhouette = {}\nmin davies_bouldin = {}\n".format(
                method,
                int(clustering_results.loc[clustering_results['method'] == method, "n_clusters"].mean()),
                int(clustering_results.loc[clustering_results['method'] == method, "n_clusters"].median()),
                int(clustering_results.loc[clustering_results['method'] == method, "noise"].median()),
                round(clustering_results.loc[clustering_results['method'] == method, "silhouette"].max(), 3),
                round(clustering_results.loc[clustering_results['method'] == method, "davies_bouldin"].min(), 3)
            ))


def plot_distributions_of_best_clustering_results(best_clustering, save_to):
    for drug in drugs:
        # filter out drugs where best clustering failed
        drug_clusters = best_clustering.loc[best_clustering['drug'] == drug, :]
        if drug_clusters['n_clusters'].sum() <= 8 or drug_clusters['n_clusters'].sum() >= 80:
            best_clustering = best_clustering.drop(best_clustering[best_clustering['drug'] == drug].index)

    best_clustering['n_clusters'] = best_clustering['n_clusters'].astype('int')
    seaborn.kdeplot(data=best_clustering, x='n_clusters', hue='method', fill=True, alpha=0.5)
    pyplot.grid()
    pyplot.tight_layout()
    pyplot.savefig(save_to + 'best_clustering_dist.pdf')


def plot_n_clusters_for_selected_drugs(best_clustering, save_to):

    selected = ['Clofarabine', 'Lenvatinib', 'Irinotecan', 'Metformin', 'Topotecan', 'Rapamycin', 'Gemcitabine', 'Paclitaxel', 'Omacetaxine']
    best_clustering = best_clustering.loc[numpy.isin(best_clustering['drug'], selected), :]

    pyplot.figure()
    seaborn.barplot(x='drug', y='n_clusters', hue='method', data=best_clustering)
    pyplot.xticks(rotation=45)
    pyplot.xlabel('')
    pyplot.ylabel('Number of clusters')
    pyplot.grid()
    pyplot.tight_layout()
    pyplot.savefig(save_to + 'best_clustering_drugs.pdf')


if __name__ == "__main__":

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'
    run_clustering_grid = False

    if run_clustering_grid:
        # run clustering over many parameter sets and save results
        path_to_drugs = 'D:\ETH\projects\morpho-learner\data\cut\\'
        collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_drugs, save_to)

    # read results of clustering
    clustering_results = pandas.read_csv(save_to + 'clusters_over_min_cluster_size.csv')
    clustering_results = clustering_results.rename(columns={'model': 'method'})

    # analyze full results
    print_statistics_on_clustering_results(clustering_results)
    plot_full_distribution_of_clusters(clustering_results, save_to)

    # find best clustering results and analyze them
    best_clustering = select_the_best_clustering_results(clustering_results)
    plot_distributions_of_best_clustering_results(best_clustering, save_to)
    plot_n_clusters_for_selected_drugs(best_clustering, save_to)

