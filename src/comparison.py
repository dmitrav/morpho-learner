import pandas, os, seaborn, numpy, umap, time, torch, scipy
from sklearn import metrics
from matplotlib import pyplot
from torchmetrics import metric
from tqdm import tqdm
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist

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

        cons_c = 0
        # calculate consistency of each cell line and then append mean
        for cell_line in cell_lines:
            cons_c += numpy.where(cluster_cells == cell_line)[0].shape[0] / cluster_indices.shape[0]
        consisten_cs.append(cons_c / len(cell_lines))

        cons_d = 0
        # calculate consistency of each drug and then append mean
        for drug in drugs:
            cons_d += numpy.where(cluster_drugs == drug)[0].shape[0] / cluster_indices.shape[0]
        consisten_ds.append(cons_d / len(drugs))

    return numpy.median(consisten_cs), numpy.median(consisten_ds)


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_images, grouping_factors, range_with_step, uid=''):
    """ Cluster the dataset over multiple parameters, evaluate results and save results as a dataframe. """

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\clustering\\'

    results = {'group_by': [], 'method': [], 'min_cluster_size': [], 'n_clusters': [],
               'noise': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [],
               'consistency_cells': [], 'consistency_drugs': []}

    for factor in tqdm(grouping_factors):
        for model in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
            for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

                transform = analysis.get_f_transform(model, setting, torch.device('cuda'))
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
                    silhouette = metrics.silhouette_score(embedding, clusters)
                    calinski_harabasz = metrics.calinski_harabasz_score(embedding, clusters)
                    davies_bouldin = metrics.davies_bouldin_score(embedding, clusters)
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


def select_the_best_clustering_results(path, grouping_factors):

    clustering_results = pandas.read_csv(path)

    # find uncorrelated metrics for further evaluations
    print("corr(silhouette, calinski_harabasz) = {}\ncorr(silhouette, davies_bouldin) = {}".format(
        scipy.stats.pearsonr(clustering_results.loc[:, 'silhouette'], clustering_results.loc[:, 'calinski_harabasz'])[0],
        scipy.stats.pearsonr(clustering_results.loc[:, 'silhouette'], clustering_results.loc[:, 'davies_bouldin'])[0]))

    best_clustering = pandas.DataFrame(columns=clustering_results.columns)
    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:
            for factor in grouping_factors:
                df = clustering_results.loc[(clustering_results['group_by'] == factor) & (clustering_results['method'] == method) & (clustering_results['setting'] == setting), :]
                # select the best results by three metrics:
                df = df[df['silhouette'] >= df['silhouette'].median()]
                df = df[df['davies_bouldin'] <= df['davies_bouldin'].median()]
                df = df[df['noise'] == df['noise'].min()]
                if df.shape[0] > 1:
                    # if there are multiple equivalent results, take more clusters
                    df = df[df['n_clusters'] == df['n_clusters'].max()]
                best_clustering = pandas.concat([best_clustering, df])

    return best_clustering


def print_statistics_on_clustering_results(path):

    # read results of clustering
    clustering_results = pandas.read_csv(path)

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:
            print(
                "{} + {}:\nmedian n_clusters = {}\nmedian noise = {}%\nmedian silhouette = {}\nmedian davies_bouldin = {}\n".format(
                    method, setting,
                    int(clustering_results.loc[(clustering_results['method'] == method) & (clustering_results['setting'] == setting), "n_clusters"].median()),
                    int(clustering_results.loc[(clustering_results['method'] == method) & (clustering_results['setting'] == setting), "noise"].median()),
                    round(clustering_results.loc[(clustering_results['method'] == method) & (clustering_results['setting'] == setting), "silhouette"].median(), 3),
                    round(clustering_results.loc[(clustering_results['method'] == method) & (clustering_results['setting'] == setting), "davies_bouldin"].median(), 3)
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

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\similarity\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'cell_line': [], 'method': [], 'setting': [], 'comparison': [],
               'euclidean': [], 'cosine': [], 'correlation': [], 'braycurtis': []}

    control_plates = ['P21_DMSO', 'N19_DMSO', 'M8_DMSO', 'L14_DMSO', 'H9_DMSO', 'E2_DMSO']

    for cell_line in tqdm(cell_lines):
        for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
            for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

                transform = analysis.get_f_transform(method, setting, torch.device('cuda'))

                encodings_MTX, _ = analysis.get_image_encodings_from_path(path_to_drugs, 'Methotrexate', transform)
                encodings_PTX, _ = analysis.get_image_encodings_from_path(path_to_drugs, 'Pemetrexed', transform)
                encodings_DMSO, _ = analysis.get_image_encodings_from_path(path_to_controls, control_plates, transform, n=10, randomize=False)

                # compare Methotrexate with Pemetrexed
                results['cell_line'].append(cell_line)
                results['method'].append(method)
                results['setting'].append(setting)
                results['comparison'].append('MTX-PTX')
                comparison = calculate_similarity_of_pair(encodings_MTX, encodings_PTX)
                results['euclidean'].append(comparison['euclidean'])
                results['cosine'].append(comparison['cosine'])
                results['correlation'].append(comparison['correlation'])
                results['braycurtis'].append(comparison['braycurtis'])

                # compare Methotrexate with DMSO (control)
                results['cell_line'].append(cell_line)
                results['method'].append(method)
                results['setting'].append(setting)
                results['comparison'].append('MTX-DMSO')
                comparison = calculate_similarity_of_pair(encodings_MTX, encodings_DMSO)
                results['euclidean'].append(comparison['euclidean'])
                results['cosine'].append(comparison['cosine'])
                results['correlation'].append(comparison['correlation'])
                results['braycurtis'].append(comparison['braycurtis'])

                # compare Pemetrexed with DMSO (control)
                results['cell_line'].append(cell_line)
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


def print_statistics_on_similarity_results(path):

    results = pandas.read_csv(path)

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

            temp = results.loc[(results['method'] == method) & (results['setting'] == setting), :]
            print("{} + {}:\nmedian euclidean = {}\nmedian cosine = {}\nmedian correlation = {}\nmedian braycurtis = {}\n".format(
                    method, setting,
                    round(temp["euclidean"].median(), 3), round(temp["cosine"].median(), 3),
                    round(temp["correlation"].median(), 3), round(temp["braycurtis"].median(), 3)
                ))


def plot_facet_grid(path):

    # TODO: I might wanna plot -log10 of similarity scores...

    attend = seaborn.load_dataset("attention").query("subject <= 16")
    g = seaborn.FacetGrid(attend, col="subject", col_wrap=4, height=2, ylim=(0, 10), margin_titles=True)
    g.map(seaborn.barplot, "solutions", "score", color=".3", ci=None)
    pyplot.show()


def calculate_classification_metrics(cell_line, codes_drugs, codes_controls, drugs_ids, controls_ids, classifier):

    # get codes
    codes_drugs = [codes_drugs[i] for i in range(len(codes_drugs)) if drugs_ids['cell_line'][i] == cell_line]
    codes_controls = [codes_controls[i] for i in range(len(codes_controls)) if controls_ids['cell_line'][i] == cell_line]
    # assign true labels
    labels_drugs = [1 for x in codes_drugs]
    labels_controls = [0 for x in codes_controls]

    # get predictions
    preds_drugs = []
    preds_controls = []
    for code in codes_drugs:
        preds_drugs.append(classifier(code))
    for code in codes_controls:
        preds_controls.append(classifier(code))

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

    results = {'cell_line': [], 'method': [], 'setting': [],
               'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'roc_auc': []}

    for method in ['unsupervised', 'self-supervised', 'weakly-supervised', 'regularized']:
        for setting in ['aug_multi_crop', 'aug_one_crop', 'no_aug_multi_crop', 'no_aug_one_crop']:

                transform = analysis.get_f_transform(method, setting, torch.device('cuda'))
                encodings_drugs, drugs_ids = analysis.get_image_encodings_from_path(path_to_drugs, "", transform)
                encodings_controls, controls_ids = analysis.get_image_encodings_from_path(path_to_controls, "", transform)

                classifier = analysis.get_f_classification(method, setting, torch.device('cuda'))
                for cell_line in picked_lines:

                    results['cell_line'].append(cell_line)
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


if __name__ == "__main__":

    path_to_test_drugs = 'D:\ETH\projects\morpho-learner\\data\\test\\drugs\\'
    path_to_test_controls = 'D:\ETH\projects\morpho-learner\\data\\test\\controls\\'

    # SIMILARITY OF KNOWN DRUGS VS CONTROLS
    compare_similarity(path_to_test_drugs, path_to_test_controls)

    similarity_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\similarity\\similarity.csv'
    print('STATISTICS ON SIMILARITY OF KNOWN DRUGS:\n\n')
    print_statistics_on_similarity_results(similarity_results_path)
    plot_facet_grid(similarity_results_path)  # TODO: plot three barplots: MTX-PTX, MTX-DMSO, PTX-DMSO

    # CLUSTERING OF TEST SET
    collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_test_drugs, [""], (30, 310, 30), uid='on_test')

    test_clustering_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\clustering\\clustering_on_test.csv'
    best_clustering = select_the_best_clustering_results(test_clustering_results_path, [""])
    print('BEST CLUSTERING OF TEST DATA:\n\n', best_clustering.to_string())

    # CLUSTERING OF PICKED CELL LINES
    collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_test_drugs, ["COLO205", "SW620", "SKMEL2"], (30, 310, 30), uid='by_cell_lines')

    cell_lines_clustering_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\clustering\\clustering_by_cell_lines.csv'
    print('STATISTICS ON CLUSTERING OF PICKED CELL LINES:\n\n')
    print_statistics_on_clustering_results(cell_lines_clustering_results_path)
    plot_facet_grid(cell_lines_clustering_results_path)  # TODO: plot three boxplots: n clusters for COLO205, SW620, SKMEL2

    # CLASSIFICATION OF DRUGS VS CONTROLS FOR PICKED CELL LINES
    collect_and_save_classification_results_for_cell_lines(path_to_test_drugs, path_to_test_controls, ["HT29", "HCT15", "ACHN"])
    test_classification_results_path = 'D:\ETH\projects\morpho-learner\\res\\comparison\\classification\\classification_for_cell_lines.csv'
    plot_facet_grid(test_classification_results_path)  # TODO: plot three barplots: accuracy for HT29, HCT15, ACHN

