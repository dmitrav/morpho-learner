import pandas, os, seaborn, numpy, umap, time, torch, scipy
from sklearn import metrics
from matplotlib import pyplot
from torchmetrics import metric
from tqdm import tqdm
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist
from pathlib import Path

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


def calculate_clustering_consistency(clusters, image_ids):
    """ This method computes consistencies of clusters,
        based on how many images of the same cell line / drug belong to the same cluster.
        That's an arbitrary experimental metric. """

    consisten_cs = []  # cell lines
    consisten_ds = []  # drugs
    n_clusters = max(clusters)+1
    for i in range(n_clusters):
        cluster_indices = numpy.where(numpy.array(clusters) == i)[0]
        cluster_cells = numpy.array(image_ids['cell_lines'])[cluster_indices]
        cluster_drugs = numpy.array(image_ids['drugs'])[cluster_indices]

        i_cons_c = []
        # calculate consistency of each cell line and then append mean
        for cell_line in numpy.unique(cluster_cells):
            i_cons_c.append(numpy.where(cluster_cells == cell_line)[0].shape[0] / cluster_indices.shape[0])
        consisten_cs.append(numpy.mean(i_cons_c) * 100)

        i_cons_d = []
        # calculate consistency of each drug and then append mean
        for drug in numpy.unique(cluster_drugs):
            i_cons_d.append(numpy.where(cluster_drugs == drug)[0].shape[0] / cluster_indices.shape[0])
        consisten_ds.append(numpy.mean(i_cons_d) * 100)

    return numpy.median(consisten_cs), numpy.median(consisten_ds)


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_images, grouping_factors, range_with_step, uid=''):
    """ Cluster the dataset over multiple parameters, evaluate results and save results as a dataframe. """

    # save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\clustering\\'
    save_to = '/Users/andreidm/ETH/projects/morpho-learner/res/comparison/clustering/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method': [], 'setting': [], 'min_cluster_size': [],
               'n_clusters': [], 'noise': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [],
               'consistency_cells': [], 'consistency_drugs': []}

    for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

            # transform = analysis.get_f_transform(model, setting, torch.device('cuda'))
            transform = analysis.get_f_transform(model, setting, torch.device('cpu'))

            for factor in tqdm(grouping_factors):
                encodings, image_ids = analysis.get_image_encodings_from_path(path_to_images, factor, transform)
                encodings = numpy.array(encodings)

                for min_cluster_size in tqdm(range(*range_with_step)):

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

                    try:
                        silhouette = metrics.silhouette_score(embedding, clusters)
                        calinski_harabasz = metrics.calinski_harabasz_score(embedding, clusters)
                        davies_bouldin = metrics.davies_bouldin_score(embedding, clusters)
                    except ValueError:
                        # single cluster
                        silhouette, calinski_harabasz, davies_bouldin = -1, -1, -1

                    consisten_c, consisten_d = calculate_clustering_consistency(clusters, image_ids)

                    results['group_by'].append(factor)
                    results['method'].append(model)
                    results['setting'].append(setting)

                    results['min_cluster_size'].append(min_cluster_size)
                    results['n_clusters'].append(n_clusters)
                    results['noise'].append(noise)
                    results['silhouette'].append(silhouette)
                    results['calinski_harabasz'].append(calinski_harabasz)
                    results['davies_bouldin'].append(davies_bouldin)
                    results['consistency_cells'].append(consisten_c)
                    results['consistency_drugs'].append(consisten_d)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'clustering_{}.csv'.format(uid), index=False)


def plot_full_distribution_of_clusters(path, save_to):

    clustering_results = pandas.read_csv(path)

    seaborn.set_theme(style="whitegrid")
    seaborn.displot(clustering_results, x='n_clusters', hue='method', multiple="dodge", binwidth=10)
    pyplot.xlabel('Number of clusters')
    pyplot.savefig(save_to + 'full_cluster_distribution.pdf')
    pyplot.close()


def select_the_best_clustering_results(path, grouping_factors, first_metric='silhouette'):

    clustering_results = pandas.read_csv(path, keep_default_na=False)

    # find uncorrelated metrics for further evaluations
    print("corr(silhouette, calinski_harabasz) = {}\ncorr(silhouette, davies_bouldin) = {}\n".format(
        round(scipy.stats.pearsonr(clustering_results.loc[:, 'silhouette'], clustering_results.loc[:, 'calinski_harabasz'])[0], 3),
        round(scipy.stats.pearsonr(clustering_results.loc[:, 'silhouette'], clustering_results.loc[:, 'davies_bouldin'])[0]), 3))

    best_clustering = pandas.DataFrame(columns=clustering_results.columns)
    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:
            for factor in grouping_factors:
                df = clustering_results.loc[(clustering_results['group_by'] == factor) & (clustering_results['method'] == method) & (clustering_results['setting'] == setting), :]
                # select the best results by three metrics:
                df = df[df[first_metric] >= df[first_metric].median()]
                df = df[df['davies_bouldin'] <= df['davies_bouldin'].median()]
                df = df[df['noise'] == df['noise'].min()]
                if df.shape[0] > 1:
                    # if there are multiple equivalent results, take more clusters
                    df = df[df['n_clusters'] == df['n_clusters'].max()]
                best_clustering = pandas.concat([best_clustering, df])

    return best_clustering


def print_statistics_on_clustering_results(clustering_results, title=""):

    print(title)

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

            print("=== {} + {} ===\n".format(method, setting))

            temp = clustering_results.loc[(clustering_results['method'] == method) & (clustering_results['setting'] == setting), :]

            print("\nmedian n_clusters = {}"
                  "\nmedian noise = {}%"
                  "\nmedian silhouette = {}"
                  "\nmedian calinski_harabasz = {}"
                  "\nmedian davies_bouldin = {}"
                  "\nmedian consistency_drugs = {}"
                  "\nmedian consistency_cells = {}\n".format(
                    int(temp["n_clusters"].median()),
                    int(temp["noise"].median()),
                    round(temp["silhouette"].median(), 3),
                    round(temp["calinski_harabasz"].median(), 3),
                    round(temp["davies_bouldin"].median(), 3),
                    round(temp["consistency_drugs"].median(), 3),
                    round(temp["consistency_cells"].median(), 3)
                ))


def plot_distributions_of_best_clustering_results(best_clustering, save_to):

    for drug in drugs:
        # filter out drugs where best clustering failed
        drug_clusters = best_clustering.loc[best_clustering['drug'] == drug, :]
        if drug_clusters['n_clusters'].sum() <= 8 or drug_clusters['n_clusters'].sum() >= 80:
            best_clustering = best_clustering.drop(best_clustering[best_clustering['drug'] == drug].index)

    best_clustering['n_clusters'] = best_clustering['n_clusters'].astype('int')
    seaborn.kdeplot(data=best_clustering, x='n_clusters', hue='method', fill=True, alpha=0.5)
    pyplot.tight_layout()
    pyplot.savefig(save_to + 'best_clustering_dist.pdf')
    pyplot.close()


def plot_n_clusters_for_selected_drugs(best_clustering, save_to):

    selected = ['Clofarabine', 'Lenvatinib', 'Irinotecan', 'Metformin', 'Topotecan', 'Rapamycin', 'Gemcitabine', 'Paclitaxel', 'Omacetaxine']
    best_clustering = best_clustering.loc[numpy.isin(best_clustering['drug'], selected), :]

    seaborn.barplot(x='drug', y='n_clusters', hue='method', data=best_clustering)
    pyplot.xticks(rotation=45)
    pyplot.xlabel('')
    pyplot.ylabel('Number of clusters')
    pyplot.tight_layout()
    pyplot.savefig(save_to + 'best_clustering_drugs.pdf')
    pyplot.close()


def compare_clustering_of_drugs():

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'
    run_clustering_grid = False

    if run_clustering_grid:
        # run clustering over many parameter sets and save results
        path = 'D:\ETH\projects\morpho-learner\data\cut\\'
        grouping_by = [drug for drug in drugs if drug not in ['DMSO', 'PBS']]
        collect_and_save_clustering_results_for_multiple_parameter_sets(path, grouping_by, (50, 310, 50), uid='by_drugs')

    # analyze full results
    path = save_to + 'clusters_over_min_cluster_size_50_300.csv'
    print_statistics_on_clustering_results(path)
    plot_full_distribution_of_clusters(path, save_to)

    # find best clustering results and analyze them
    best_clustering = select_the_best_clustering_results(path, grouping_by)
    plot_distributions_of_best_clustering_results(best_clustering, save_to)
    plot_n_clusters_for_selected_drugs(best_clustering, save_to)


def calculate_similarity_of_pair(codes_A, codes_B):

    d_2 = []
    d_cos = []
    d_corr = []
    d_bray = []

    for code_a in codes_A:
        for code_b in codes_B:

            d_2.append(pdist([code_a, code_b], metric='euclidean'))
            d_cos.append(pdist([code_a, code_b], metric='cosine'))
            d_corr.append(pdist([code_a, code_b], metric='correlation'))
            d_bray.append(pdist([code_a, code_b], metric='braycurtis'))

    res = {
        'euclidean': numpy.median(d_2),
        'cosine': numpy.median(d_cos),
        'correlation': numpy.median(d_corr),
        'braycurtis': numpy.median(d_bray),
    }

    return res


def compare_similarity(path_to_drugs, path_to_controls):

    # save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\similarity\\'
    save_to = '/Users/andreidm/ETH/projects/morpho-learner/res/comparison/similarity/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method': [], 'setting': [], 'comparison': [],
               'euclidean': [], 'cosine': [], 'correlation': [], 'braycurtis': []}

    control_plates = ['P21_DMSO', 'N19_DMSO', 'M8_DMSO', 'L14_DMSO', 'H9_DMSO', 'E2_DMSO']

    for cell_line in tqdm(cell_lines):
        for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
            for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

                # transform = analysis.get_f_transform(method, setting, torch.device('cuda'))
                transform = analysis.get_f_transform(method, setting, torch.device('cpu'))

                encodings_MTX, _ = analysis.get_image_encodings_from_path(path_to_drugs, ['Methotrexate', cell_line], transform)
                encodings_PTX, _ = analysis.get_image_encodings_from_path(path_to_drugs, ['Pemetrexed', cell_line], transform)

                encodings_DMSO = []
                for plate in control_plates:
                    encodings, _ = analysis.get_image_encodings_from_path(path_to_controls, [plate, cell_line], transform, n=10, randomize=False)
                    encodings_DMSO.extend(encodings)

                # compare Methotrexate with Pemetrexed
                results['group_by'].append(cell_line)
                results['method'].append(method)
                results['setting'].append(setting)
                results['comparison'].append('MTX-PTX')
                comparison = calculate_similarity_of_pair(encodings_MTX, encodings_PTX)
                results['euclidean'].append(comparison['euclidean'])
                results['cosine'].append(comparison['cosine'])
                results['correlation'].append(comparison['correlation'])
                results['braycurtis'].append(comparison['braycurtis'])

                # compare Methotrexate with DMSO (control)
                results['group_by'].append(cell_line)
                results['method'].append(method)
                results['setting'].append(setting)
                results['comparison'].append('MTX-DMSO')
                comparison = calculate_similarity_of_pair(encodings_MTX, encodings_DMSO)
                results['euclidean'].append(comparison['euclidean'])
                results['cosine'].append(comparison['cosine'])
                results['correlation'].append(comparison['correlation'])
                results['braycurtis'].append(comparison['braycurtis'])

                # compare Pemetrexed with DMSO (control)
                results['group_by'].append(cell_line)
                results['method'].append(method)
                results['setting'].append(setting)
                results['comparison'].append('PTX-DMSO')
                comparison = calculate_similarity_of_pair(encodings_PTX, encodings_DMSO)
                results['euclidean'].append(comparison['euclidean'])
                results['cosine'].append(comparison['cosine'])
                results['correlation'].append(comparison['correlation'])
                results['braycurtis'].append(comparison['braycurtis'])

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'similarity.csv', index=False)


def print_statistics_on_similarity_results(results, title=""):

    print(title)

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

            temp = results.loc[(results['method'] == method) & (results['setting'] == setting), :]

            mtx_ptx = temp.loc[temp['comparison'] == 'MTX-PTX', :]
            mtx_dmso = temp.loc[temp['comparison'] == 'MTX-DMSO', :]
            ptx_dmso = temp.loc[temp['comparison'] == 'PTX-DMSO', :]

            print("=== {} + {} ===\n".format(method, setting))

            print("MTX-PTX:\nmedian euclidean = {}\nmedian cosine = {}\nmedian correlation = {}\nmedian braycurtis = {}\n".format(
                    round(mtx_ptx["euclidean"].median(), 3), round(mtx_ptx["cosine"].median(), 3),
                    round(mtx_ptx["correlation"].median(), 3), round(mtx_ptx["braycurtis"].median(), 3)
                ))

            print("MTX-DMSO:\nmedian euclidean = {}\nmedian cosine = {}\nmedian correlation = {}\nmedian braycurtis = {}\n".format(
                    round(mtx_dmso["euclidean"].median(), 3), round(mtx_dmso["cosine"].median(), 3),
                    round(mtx_dmso["correlation"].median(), 3), round(mtx_dmso["braycurtis"].median(), 3)
                ))

            print("PTX-DMSO:\nmedian euclidean = {}\nmedian cosine = {}\nmedian correlation = {}\nmedian braycurtis = {}\n".format(
                    round(ptx_dmso["euclidean"].median(), 3), round(ptx_dmso["cosine"].median(), 3),
                    round(ptx_dmso["correlation"].median(), 3), round(ptx_dmso["braycurtis"].median(), 3)
                ))


def plot_facet_grid(data, x_variable, y_variable, hue, ci=None, plot_title="", yticks=None):

    data['method+setting'] = data['method'] + '\n' + data['setting']

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'

    seaborn.set(font_scale=0.5)
    g = seaborn.FacetGrid(data, col="method+setting", hue=hue, col_wrap=4, height=1, margin_titles=True)
    g.map(seaborn.barplot, x_variable, y_variable, order=list(data[hue].unique()), ci=ci)
    g.set_titles("{col_name}")
    if yticks is not None:
        g.set(yticks=yticks)
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + '{}.pdf'.format(plot_title))


def calculate_classification_metrics(cell_line, codes_drugs, codes_controls, drugs_ids, controls_ids, classifier, device=torch.device('cuda')):

    # get codes
    codes_drugs = [codes_drugs[i] for i in range(len(codes_drugs)) if drugs_ids['cell_lines'][i] == cell_line]
    codes_controls = [codes_controls[i] for i in range(len(codes_controls)) if controls_ids['cell_lines'][i] == cell_line]
    # assign true labels
    labels_drugs = [1 for x in codes_drugs]
    labels_controls = [0 for x in codes_controls]

    # get predictions
    preds_drugs = []
    preds_controls = []
    for code in codes_drugs:
        preds_drugs.append(int(classifier(code).argmax(-1)))
    for code in codes_controls:
        preds_controls.append(int(classifier(code).argmax(-1)))

    # calculate metrics
    acc = metrics.accuracy_score([*labels_drugs, *labels_controls], [*preds_drugs, *preds_controls])
    f1 = metrics.f1_score([*labels_drugs, *labels_controls], [*preds_drugs, *preds_controls])
    rec = metrics.recall_score([*labels_drugs, *labels_controls], [*preds_drugs, *preds_controls])
    prec = metrics.precision_score([*labels_drugs, *labels_controls], [*preds_drugs, *preds_controls])
    roc_auc = metrics.roc_auc_score([*labels_drugs, *labels_controls], [*preds_drugs, *preds_controls])

    return acc, f1, rec, prec, roc_auc


def collect_and_save_classification_results_for_cell_lines(path_to_drugs, path_to_controls, picked_lines):

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\classification\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method': [], 'setting': [],
               'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'roc_auc': []}

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

                transform = analysis.get_f_transform(method, setting, torch.device('cuda'))
                encodings_drugs, drugs_ids = analysis.get_image_encodings_from_path(path_to_drugs, "", transform)
                encodings_controls, controls_ids = analysis.get_image_encodings_from_path(path_to_controls, "", transform)

                classifier = analysis.get_f_classification(method, setting, torch.device('cuda'))
                for cell_line in picked_lines:

                    results['group_by'].append(cell_line)
                    results['method'].append(method)
                    results['setting'].append(setting)

                    acc, f1, rec, prec, roc_auc = calculate_classification_metrics(cell_line, encodings_drugs, encodings_controls, drugs_ids, controls_ids, classifier)
                    results['accuracy'].append(acc)
                    results['f1'].append(f1)
                    results['recall'].append(rec)
                    results['precision'].append(prec)
                    results['roc_auc'].append(roc_auc)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'classification_for_cell_lines.csv', index=False)


def find_and_print_best_classification():

    path = Path('D:\ETH\projects\morpho-learner\\res\linear_evaluation\\')

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

            print('===', method, '+', setting, '===\n')

            res = []
            for set_folder in os.listdir(path / method / setting):
                if not str(set_folder).endswith('.torch'):
                    history = pandas.read_csv(path / method / setting / set_folder / 'history.csv', index_col='epoch')
                    train_metric_sum = history.iloc[:, 0] + history.iloc[:, 2] + history.iloc[:, 4] + history.iloc[:, 6] + history.iloc[:, 8]
                    val_metric_sum = history.iloc[:, 1] + history.iloc[:, 3] + history.iloc[:, 5] + history.iloc[:, 7] + history.iloc[:, 9]
                    res.append((train_metric_sum.max(), val_metric_sum.max(), val_metric_sum.idxmax()))

            max_res = max(res, key=lambda x: x[1])
            print(os.listdir(path / method / setting)[res.index(max_res)])
            print(max_res, '\n')


if __name__ == "__main__":

    path_to_test_drugs = 'D:\ETH\projects\morpho-learner\\data\\test\\drugs\\'
    path_to_test_controls = 'D:\ETH\projects\morpho-learner\\data\\test\\controls\\'

    # SIMILARITY OF KNOWN DRUGS VS CONTROLS
    # compare_similarity(path_to_test_drugs, path_to_test_controls)

    similarity_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\similarity\\similarity.csv'
    # sim_data = pandas.read_csv(similarity_results_path)
    # cl_subset = 'M14'
    # sim_data = sim_data.loc[sim_data['group_by'] == cl_subset, :]
    # print_statistics_on_similarity_results(sim_data, title='STATISTICS ON SIMILARITY OF KNOWN DRUGS:')
    # plot_facet_grid(sim_data, 'comparison', 'euclidean', 'comparison', ci='sd', plot_title='similarity_of_drugs_euclidean_{}'.format(cl_subset))

    # CLUSTERING OF CELL LINES
    # collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_test_drugs, cell_lines, (10, 160, 10), uid='by_cell_lines')
    # cell_lines_clustering_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\clustering\\clustering_by_cell_lines.csv'

    # cl_clust_data = pandas.read_csv(cell_lines_clustering_results_path)
    # print_statistics_on_clustering_results(cl_clust_data, title='STATISTICS ON CLUSTERING OF CELL LINES:')
    # # now subset to COLO205, SW620, SKMEL2
    # cl_clust_data = cl_clust_data.loc[(cl_clust_data['group_by'] == 'COLO205') | (cl_clust_data['group_by'] == 'SW620') | (cl_clust_data['group_by'] == 'SKMEL2'), :]
    # plot_facet_grid(cl_clust_data, 'group_by', 'n_clusters', 'group_by', ci=80, plot_title="clustering_picked_cell_lines")
    # plot_facet_grid(cl_clust_data, 'group_by', 'silhouette', 'group_by', ci=80, plot_title="silhouette_picked_cell_lines")
    # plot_facet_grid(cl_clust_data, 'group_by', 'calinski_harabasz', 'group_by', ci=80, plot_title="calinski_picked_cell_lines")
    # plot_facet_grid(cl_clust_data, 'group_by', 'davies_bouldin', 'group_by', ci=80, plot_title="davies_picked_cell_lines")
    # plot_facet_grid(cl_clust_data, 'group_by', 'consistency_drugs', 'group_by', ci=80, plot_title="consistency_picked_cell_lines")
    # plot_facet_grid(cl_clust_data, 'group_by', 'noise', 'group_by', ci=80, plot_title="noise_picked_cell_lines")
    #
    # # CLUSTERING OF DRUGS
    # picked_drugs = [d for d in drugs if d not in ['PBS', 'DMSO']]
    # collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_test_drugs, picked_drugs, (10, 160, 10), uid='by_drugs')
    # drugs_clustering_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\clustering\\clustering_by_drugs.csv'
    #
    # d_clust_data = pandas.read_csv(drugs_clustering_results_path)  # now subset to Gemcitabine, Cladribine, Irinotecan
    # print_statistics_on_clustering_results(d_clust_data, title='STATISTICS ON CLUSTERING OF DRUGS:')
    # # now subset to Gemcitabine, Cladribine, Irinotecan
    # d_clust_data = d_clust_data.loc[(d_clust_data['group_by2'] == 'Gemcitabine') | (d_clust_data['group_by'] == 'Cladribine') | (d_clust_data['group_by'] == 'Irinotecan'), :]
    # plot_facet_grid(d_clust_data, 'group_by', 'n_clusters', 'group_by', ci=80, plot_title="clustering_picked_drugs")
    # plot_facet_grid(d_clust_data, 'group_by', 'silhouette', 'group_by', ci=80, plot_title="silhouette_picked_drugs")
    # plot_facet_grid(d_clust_data, 'group_by', 'calinski_harabasz', 'group_by', ci=80, plot_title="calinski_picked_drugs")
    # plot_facet_grid(d_clust_data, 'group_by', 'davies_bouldin', 'group_by', ci=80, plot_title="davies_picked_drugs")
    # plot_facet_grid(d_clust_data, 'group_by', 'consistency_cells', 'group_by', ci=80, plot_title="consistency_picked_drugs")
    # plot_facet_grid(d_clust_data, 'group_by', 'noise', 'group_by', ci=80, plot_title="noise_picked_drugs")

    # CLASSIFICATION OF DRUGS VS CONTROLS FOR PICKED CELL LINES
    find_and_print_best_classification()

    path_to_full_drugs = 'D:\ETH\projects\morpho-learner\\data\\full\\drugs\\'
    path_to_full_controls = 'D:\ETH\projects\morpho-learner\\data\\full\\controls\\'

    # collect_and_save_classification_results_for_cell_lines(path_to_full_drugs, path_to_full_controls, cell_lines)
    # test_classification_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\classification\\classification_for_cell_lines.csv'
    # class_data = pandas.read_csv(test_classification_results_path)
    #
    # for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
    #     for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:
    #
    #         print('=== {} + {} ==='.format(method, setting))
    #
    #         d = class_data.loc[(class_data['method'] == method) & (class_data['setting'] == setting), :]
    #         print('mean acc = {}\n'
    #               'mean f1 = {}\n'
    #               'mean recall = {}\n'
    #               'mean precision = {}\n'
    #               'mean roc_auc = {}\n'.format(d['accuracy'].median(),
    #                                          d['f1'].median(),
    #                                          d['recall'].median(),
    #                                          d['precision'].median(),
    #                                          d['roc_auc'].median()))


    # yticks = [0.2, 0.4, 0.6, 0.8]
    # plot_facet_grid(class_data, 'group_by', 'roc_auc', 'group_by', ci=80, plot_title='roc_auc', yticks=yticks)
    # plot_facet_grid(class_data, 'group_by', 'f1', 'group_by', ci=80, plot_title='f1', yticks=[0.1, 0.3, 0.5, 0.7])
    # plot_facet_grid(class_data, 'group_by', 'accuracy', 'group_by', ci=80, plot_title='accuracy', yticks=yticks)
    # plot_facet_grid(class_data, 'group_by', 'precision', 'group_by', ci=80, plot_title='precision', yticks=yticks)
    # plot_facet_grid(class_data, 'group_by', 'recall', 'group_by', ci=80, plot_title='recall', yticks=yticks)

