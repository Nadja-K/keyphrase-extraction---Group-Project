import colorsys

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import sklearn.cluster
import numpy as np
from typing import Callable
from abc import ABCMeta, abstractmethod
import scipy.spatial.distance as ssd
import time

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

    def calc_clusters(self, num_clusters, cluster_features, labels):
        # Anm.: die cluster_feature matrix ist NICHT immer eine symmetrische matrix (die Diagonale kann Werte != 0 haben
        # die Warnung dazu kann also ignoriert werden.
        cluster_features = 1. / (cluster_features + 0.001)
        print(cluster_features)
        print(labels)
        linked = linkage(cluster_features, self.method)
        clusters = fcluster(linked, num_clusters, criterion='maxclust')

        def get_N_HexCol(N=5):
            HSV_tuples = [(x * 1.0 / N*2, 1.0, 1.0) for x in range(N)]
            hex_out = []
            for rgb in HSV_tuples:
                rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
                hex_out.append('#%02x%02x%02x' % tuple(rgb))
            return hex_out

        cluster_colors = get_N_HexCol(len(set(clusters)))
        ################
        dflt_col = "#404040"  # Unclustered gray
        D_leaf_colors = dict()
        for index, cluster in enumerate(clusters):
            label = labels[index]
            D_leaf_colors[label] = cluster_colors[cluster-1]
        print(D_leaf_colors)
        link_cols = {}
        for i, i12 in enumerate(linked[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linked) else D_leaf_colors[labels[x]] for x in i12)
            link_cols[i + 1 + len(linked)] = c1 if c1 == c2 else dflt_col
        ###########

        fig, axes = plt.subplots(1, 1, figsize=(20, 10))
        dendrogram(linked,
                   labels=labels,
                   # truncate_mode='lastp',
                   # p=30,
                   orientation='top',
                   leaf_font_size=12,
                   #distance_sort='descending',
                   #show_contracted=True,
                   #show_leaf_counts=True),
                   ax=axes,
                   link_color_func=lambda k: link_cols[k]
                   )

       # for tick in axes.xaxis.get_major_ticks():
       #     label = tick.label._text
       #     index = labels.index(label)
       #     cluster = clusters[index]
       #     tick.label.set_color(colors[cluster-1])  # set the color
        plt.savefig('cluster_graphs/' + str(time.time()) + ".png")
        print(num_clusters)
        print(cluster_features.shape)
        print(len(labels))
        print(labels, clusters)
        input("h")
        return clusters


class SpectralClustering(Clustering):
    def calc_clusters(self, num_clusters, cluster_features, labels):
        clusters = sklearn.cluster.SpectralClustering(n_clusters=num_clusters).fit(cluster_features)
        return clusters.labels_


class AffinityPropagation(Clustering):
    def calc_clusters(self):
        # FIXME
        pass

