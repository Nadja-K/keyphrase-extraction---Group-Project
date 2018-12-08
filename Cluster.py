from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable
from abc import ABCMeta, abstractmethod
import scipy.spatial.distance as ssd


def euclid_dist(cluster_features, mean_cluster_features):
    return np.linalg.norm(cluster_features - mean_cluster_features)


class Clustering(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        pass

    def get_exemplar_terms(self, clusters, cluster_features, dist_func: Callable = euclid_dist):
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

        return cluster_exemplar_terms


# Hierarchical Clustering (good explanation: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-
# dendrogram-tutorial/)
class HierarchicalClustering(Clustering):
    def __init__(self, **kwargs):
        self.method = kwargs.get('method', 'ward')

    def calc_clusters(self, num_clusters, cluster_features):
        # Anm.: die cluster_feature matrix ist NICHT immer eine symmetrische matrix (die Diagonale kann Werte != 0 haben
        # die Warnung dazu kann also ignoriert werden.
        linked = linkage(cluster_features, self.method)

        # # FIXME: remove the plot stuff
        # plt.figure(figsize=(10, 7))
        # dendrogram(linked,
        #            truncate_mode='lastp',
        #            p=30,
        #            orientation='top',
        #            distance_sort='descending',
        #            show_contracted=True,
        #            show_leaf_counts=True)
        # plt.show()

        clusters = fcluster(linked, num_clusters, criterion='maxclust')
        return clusters


class SpectralClustering(Clustering):
    def calc_clusters(self):
        # FIXME
        pass


class AffinityPropagation(Clustering):
    def calc_clusters(self):
        # FIXME
        pass

