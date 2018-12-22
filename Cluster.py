import colorsys
import os

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import sklearn.cluster
import numpy as np
from typing import Callable
from abc import ABCMeta, abstractmethod
import scipy.spatial.distance as ssd
import seaborn as sns
from sklearn.decomposition import PCA
import time
import logging

from ClusterFeatureCalculator import CooccurrenceClusterFeature, WordEmbeddingsClusterFeature, PPMIClusterFeature


def euclid_dist(cluster_features, mean_cluster_features):
    return np.linalg.norm(cluster_features - mean_cluster_features)


class Clustering(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        pass

    def _transform_to_distance_matrix(self, cluster_features):
        # Adjust the feature matrix so that you get a distance matrix!
        if self.cluster_feature_calculator in [CooccurrenceClusterFeature, PPMIClusterFeature]:
            cluster_features = 1. / (cluster_features + 0.1)

        elif self.cluster_feature_calculator is WordEmbeddingsClusterFeature:
            # Calculate a cosine distance matrix
            if self.word_embedding_comp_func is sklearn.metrics.pairwise.cosine_similarity:
                cluster_features = 1 - cluster_features
            elif self.word_embedding_comp_func is np.dot:
                # FIXME: ensure that this is the correct way to turn this into a distance matrix
                cluster_features = 1. / (cluster_features + 0.1)
            else:
                logging.warning("The input matrix for the clustering was not re-calculated into a distance matrix. "
                                "Make sure this is actually what you want.")

        return cluster_features

    def _create_heatmap(self, data, labels, filename):
        labels = list(labels)

        plt.subplots(figsize=(16, 11))
        plot = sns.heatmap(data, xticklabels=labels, yticklabels=labels, square=True)
        plot.xaxis.set_ticks_position('top')
        plot.xaxis.set_tick_params(rotation=90, labelsize='large')
        plot.yaxis.set_tick_params(labelsize='large')
        fig = plot.get_figure()

        plt.title('Heatmap Feature Visualization', fontdict={'fontsize': 20})
        plt.tight_layout()
        fig.savefig('heatmap_graphs/' + os.path.basename(filename).split('.')[0] + ".png")
        plt.close()

    def _create_simple_cluster_visualization(self, data, labels, clusters, filename):
        labels = list(labels)
        clusters = list(clusters)
        big_clusters = set([i for i in clusters if clusters.count(i) > 1])
        num_big_clusters = len(big_clusters)
        big_clusters = list(big_clusters)

        cmap = plt.cm.get_cmap("gist_rainbow")
        color_mapping = []
        for cluster in clusters:
            if clusters.count(cluster) == 1:
                color_mapping.append((0.6, 0.6, 0.6, 1.0))
            else:
                color_mapping.append(cmap(float(big_clusters.index(cluster)) / num_big_clusters))
        # Transform the data into 2 dimensions
        pca = PCA(n_components=2).fit(data)
        pca_2d = pca.transform(data)
        x = pca_2d[:, 0]
        y = pca_2d[:, 1]

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(x, y, s=45, c=color_mapping)#[cmap(float(i) /num_clusters) for i in clusters])#c=clusters)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

        plt.title('Simplified Cluster Visualization', fontdict={'fontsize': 20})
        plt.tight_layout()
        fig.savefig('simple_cluster_graphs/' + os.path.basename(filename).split('.')[0] + ".png")
        plt.close()

    def get_exemplar_terms(self, clusters, cluster_features, terms, dist_func: Callable = euclid_dist):
        terms = list(terms)
        cluster_exemplar_terms = dict()

        # Derive the centroids of each cluster based on the euclidean distance
        for index, cluster in enumerate(clusters):
            if cluster not in cluster_exemplar_terms:
                cluster_exemplar_terms[cluster] = {
                    'sum': np.zeros(len(cluster_features[index])),
                    'num_samples': 0,
                    'centroid': np.zeros(len(cluster_features[index])),
                    'centroid_dist': 99999,
                    'centroid_index': -1,
                    'mean': 0
                }
            cluster_exemplar_terms[cluster]['sum'] += cluster_features[index]
            cluster_exemplar_terms[cluster]['num_samples'] += 1

        # Calculate the mean value for each cluster
        for cluster, _ in cluster_exemplar_terms.items():
            cluster_exemplar_terms[cluster]['mean'] = cluster_exemplar_terms[cluster]['sum'] / \
                                                      cluster_exemplar_terms[cluster]['num_samples']

        # Get the centroid for each cluster based on the euclidean distance to the mean
        for index, cluster in enumerate(clusters):
            dist = dist_func(cluster_features[index], cluster_exemplar_terms[cluster]['mean'])

            if dist < cluster_exemplar_terms[cluster]['centroid_dist']:
                cluster_exemplar_terms[cluster]['centroid'] = cluster_features[index]
                cluster_exemplar_terms[cluster]['centroid_dist'] = dist
                cluster_exemplar_terms[cluster]['centroid_index'] = index
                cluster_exemplar_terms[cluster]['term'] = terms[index]

        return cluster_exemplar_terms

    def get_random_exemplar_terms(self, num_clusters, terms):
        terms = list(terms)
        cluster_exemplar_terms = dict()

        selected_exemplar_terms = np.random.choice(terms, size=num_clusters, replace=False)
        for index in range(num_clusters):
            cluster_exemplar_terms[index+1] = {
                'sum': np.array([]),
                'num_samples': 0,
                'centroid': np.array([]),
                'centroid_dist': 99999,
                'centroid_index': terms.index(selected_exemplar_terms[index]),
                'mean': 0,
                'term': selected_exemplar_terms[index]
            }

        return cluster_exemplar_terms

def get_N_HexCol(N=5):
    HSV_tuples = [(x * 1.0 / N*2, 1.0, 1.0) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out


# Hierarchical Clustering (good explanation: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-
# dendrogram-tutorial/)
class HierarchicalClustering(Clustering):
    def __init__(self, **kwargs):
        self.method = kwargs.get('method', 'ward')
        self.cluster_feature_calculator = kwargs.get('cluster_feature_calculator', CooccurrenceClusterFeature)
        self.word_embedding_comp_func = kwargs.get('word_embedding_comp_func', sklearn.metrics.pairwise.cosine_similarity)

    def _create_dendogram(self, linked, labels, clusters, filename):
        # Create hex colors for every cluster
        cluster_colors = get_N_HexCol(len(set(clusters)))
        dflt_col = "#404040"  # Unclustered gray

        # Create a dictionary for all leaves (words) and their corresponding cluster color
        D_leaf_colors = dict()
        for index, cluster in enumerate(clusters):
            label = labels[index]
            D_leaf_colors[label] = cluster_colors[cluster - 1]

        # Create a dict to color the links between the samples of each cluster
        link_cols = {}
        for i, i12 in enumerate(linked[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linked) else D_leaf_colors[labels[x]] for x in i12)
            link_cols[i + 1 + len(linked)] = c1 if c1 == c2 else dflt_col
            for x in i12:
                if c1 != c2 and x < len(linked):
                    D_leaf_colors[labels[x]] = dflt_col

        # Create the actual dendogram with the calculated colors
        fig, axes = plt.subplots(1, 1, figsize=(20, 5))
        dendrogram(linked,
                   labels=labels,
                   orientation='top',
                   leaf_font_size=12,
                   # distance_sort='descending',
                   ax=axes,
                   link_color_func=lambda k: link_cols[k],
                   # leaf_rotation=60.0
                   )

        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_color(D_leaf_colors[tick.label._text])

        plt.title('Dendogram Cluster Visualization', fontdict={'fontsize': 20})
        plt.tight_layout()
        plt.savefig('cluster_graphs/' + os.path.basename(filename).split('.')[0] + ".png")
        plt.close()

    def calc_clusters(self, num_clusters, cluster_features, labels, filename=str(time.time()), draw_graphs=False):
        labels = list(labels)
        # Anm.: die cluster_feature matrix ist NICHT immer eine symmetrische matrix (die Diagonale kann Werte != 0 haben
        # die Warnung dazu kann also ignoriert werden.

        if draw_graphs is True:
            print("Creating similarity heatmap feature visualization for %s" % os.path.basename(filename))
            self._create_heatmap(cluster_features, labels, filename + "_hierarchical_similarity")

        # Adjust the feature matrix so that you get a distance matrix!
        cluster_features = self._transform_to_distance_matrix(cluster_features)

        if draw_graphs is True:
            print("Creating distance heatmap feature visualization for %s" % os.path.basename(filename))
            self._create_heatmap(cluster_features, labels, filename + "_hierarchical_distance")

        linked = linkage(cluster_features, self.method)
        clusters = fcluster(linked, num_clusters, criterion='maxclust')

        if draw_graphs is True:
            print("Creating dendogram for %s" % os.path.basename(filename))
            self._create_dendogram(linked, labels, clusters, filename)

            print("Creating simplified cluster visualization for %s" % os.path.basename(filename))
            self._create_simple_cluster_visualization(cluster_features, labels, clusters, filename + "_hierarchical")

        return clusters


class SpectralClustering(Clustering):
    def __init__(self, **kwargs):
        self.cluster_feature_calculator = kwargs.get('cluster_feature_calculator', CooccurrenceClusterFeature)
        self.word_embedding_comp_func = kwargs.get('word_embedding_comp_func', sklearn.metrics.pairwise.cosine_similarity)

    def calc_clusters(self, num_clusters, cluster_features, labels, filename=str(time.time()), draw_graphs=False):
        labels = list(labels)

        if draw_graphs is True:
            print("Creating similarity heatmap feature visualization for %s" % os.path.basename(filename))
            self._create_heatmap(cluster_features, labels, filename + "_spectral_similarity")

        # Adjust the feature matrix so that you get a distance matrix!
        cluster_features = self._transform_to_distance_matrix(cluster_features)

        if draw_graphs is True:
            print("Creating distance heatmap feature visualization for %s" % os.path.basename(filename))
            self._create_heatmap(cluster_features, labels, filename + "_spectral_distance")

        # Calculate an affinity matrix from the distance matrix
        cluster_features = np.exp(-0.1 * cluster_features / (cluster_features.std()))
        if draw_graphs is True:
            print("Creating affinity heatmap feature visualization for %s" % os.path.basename(filename))
            self._create_heatmap(cluster_features, labels, filename + "_spectral_affinity")

        clusters = sklearn.cluster.SpectralClustering(n_clusters=num_clusters, affinity='precomputed', n_jobs=-1).fit(cluster_features )
        if draw_graphs is True:
            print("Creating simplified cluster visualization for %s" % os.path.basename(filename))
            self._create_simple_cluster_visualization(cluster_features, labels, clusters.labels_, filename + "_spectral")

        return clusters.labels_


class AffinityPropagation(Clustering):
    def calc_clusters(self):
        # FIXME
        pass

