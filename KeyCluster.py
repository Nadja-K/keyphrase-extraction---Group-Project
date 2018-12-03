import string
import numpy as np
import time
from pke.base import LoadFile
from pke.data_structures import Candidate
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import re
from collections import defaultdict

class KeyCluster(LoadFile):
    def __init__(self):
        super(KeyCluster, self).__init__()
        self.tokens = None
        self.cooccurrence_matrix = None

    def candidate_selection(self, stoplist=None, **kwargs):
        # tokenization -> is done during the dataset preprocessing ?

        # no n_gram selection, instead single-word terms are used as candidates at the beginning
        self.ngram_selection(n=1)

        # Filter out punctuation, we need all candidates (including stopwords) later
        self.candidate_filtering(stoplist=list(string.punctuation), mininum_length=1, mininum_word_size=1, only_alphanum=False)#, pos_blacklist=["POS", "."])
        self.tokens = self.candidates.copy()

        # remove stop words
        if stoplist is None:
            stoplist = self.stoplist

        # FIXME: ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] <-- wird in anderen algorithmen verwendet, wofür steht das???
        self.candidate_filtering(stoplist=list(string.punctuation) + ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] + stoplist)

    def calc_cooccurrence_matrix(self, window=2):
        # FIXME: bessere/schnellere alternative???
        tokens = list(self.tokens)

        start_time = time.time()
        data = np.zeros((len(tokens), len(tokens)))
        for sentence in list(self.sentences):
            words = sentence.stems
            pos_tag = sentence.pos

            # Remove words/symbols that don't appear in the punctuation filtered tokens list
            for index in sorted([i for i, x in enumerate(words) if x not in tokens], reverse=True):
                pos_tag.pop(index)
                words.pop(index)

            for pos in range(len(words)):
                start = pos - window
                end = pos + window + 1
                word_index = tokens.index(words[pos])

                if start < 0:
                    start = 0

                for word in words[start:pos] + words[pos+1:end]:
                    data[word_index][tokens.index(word)] += 1

        # Alternative way to calculate the co-occurrence matrix that doesn't work right now because the offset is not
        # exactly true if punctuation is supposed to be ignored (punctuation offset counts here but it should not).
        # The above approach doesn't have this problem.
        # data = np.zeros((len(tokens), len(tokens)))
        # for pos1, word1 in enumerate(tokens):
        #     word1_offsets = self.tokens[word1].offsets
        #     word1_sentence_ids = self.tokens[word1].sentence_ids
        #
        #     for pos2, word2 in enumerate(tokens):
        #         if pos2 == pos1:
        #             continue
        #         word2_offsets = self.tokens[word2].offsets
        #         word2_sentence_ids = self.tokens[word2].sentence_ids
        #
        #         # print(word1, word1_offsets, word2, word2_offsets)
        #
        #         for offset1, sentence1_id in zip(word1_offsets, word1_sentence_ids):
        #             for offset2, sentence2_id in zip(word2_offsets, word2_sentence_ids):
        #                 if abs(offset2 - offset1) <= window and sentence1_id == sentence2_id:
        #                     data[pos1][pos2] += 1

        end_time = time.time()
        # print(end_time-start_time)
        # print(data)
        self.cooccurrence_matrix = data

    def candidate_weighting(self, stoplist=None, window=2):
        # Calculating term relatedness
            # Cooccurrence-based Term-Relatedness
        self.calc_cooccurrence_matrix(window=window)

            # Wikipedia-based Term-Relatedness
        # FIXME

        # Term clustering
            # Hierarchical Clustering (good explanation: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)
        linked = linkage(self.cooccurrence_matrix, 'ward')

        # FIXME: remove the plot stuff
        plt.figure(figsize=(10, 7))
        dendrogram(linked,
                   truncate_mode='lastp',
                   p=30,
                   orientation='top',
                   distance_sort='descending',
                   show_contracted=True,
                   show_leaf_counts=True)
        plt.show()

        num_clusters = int(1./3. * len(self.tokens))
        clusters = fcluster(linked, num_clusters, criterion='maxclust')

        # Spectral Clustering
            # FIXME

        # Affinity Propagation
            # FIXME

        #####################
        # Get exemplar terms
        #####################
        # Derive the centroids of each cluster based on the euclidean distance
        cluster_distances = dict()
        for index, cluster in enumerate(clusters):
            if cluster not in cluster_distances:
                cluster_distances[cluster] = dict()
                cluster_distances[cluster]['sum'] = np.zeros(len(self.cooccurrence_matrix[index]))
                cluster_distances[cluster]['samples'] = 0
                cluster_distances[cluster]['centroid'] = np.zeros(len(self.cooccurrence_matrix[index]))
                cluster_distances[cluster]['centroid_dist'] = 99999
                cluster_distances[cluster]['centroid_index'] = -1
            cluster_distances[cluster]['sum'] += self.cooccurrence_matrix[index]
            cluster_distances[cluster]['samples'] += 1

        for cluster, sum in cluster_distances.items():
            cluster_distances[cluster]['mean'] = cluster_distances[cluster]['sum'] / cluster_distances[cluster]['samples']
            # print(cluster_distances[cluster]['mean'])

        tokens = list(self.tokens)
        if stoplist is None:
            stoplist = self.stoplist

        for index, cluster in enumerate(clusters):
            # Stopword filtering -> Ignore all words that are stopwords, some clusters are ignored because of this
            if tokens[index] in list(string.punctuation) + stoplist:
                continue

            dist = np.linalg.norm(self.cooccurrence_matrix[index] - cluster_distances[cluster]['mean'])

            if dist < cluster_distances[cluster]['centroid_dist']:
                cluster_distances[cluster]['centroid'] = self.cooccurrence_matrix[index]
                cluster_distances[cluster]['centroid_dist'] = dist
                cluster_distances[cluster]['centroid_index'] = index

        # Remove clusters that are now empty because  of the stopword filtering
        for cluster, val in sorted(cluster_distances.items()):
            if cluster_distances[cluster]['centroid_index'] == -1:
                del cluster_distances[cluster]

        # Access exemplar terms
        # for key, val in cluster_distances.items():
        #     print(key, tokens[val['centroid_index']], val['centroid_dist'])

        ##########################################
        # FIXME: create candidate keyphrases?
        ##########################################
        candidate_keyphrases = dict()
        for sentence_id, sentence in enumerate(list(self.sentences)):

            pos_tags = ""
            # FIXME: might not be correct yet? NNS NNP?
            for pos in sentence.pos:
                if pos == 'NOUN':
                    pos_tags = pos_tags + "n"
                elif pos == 'ADJ':
                    pos_tags = pos_tags + "j"
                else:
                    pos_tags = pos_tags + "x"

            # Use the regex from the original paper (JJ)*(NN|NNS|NNP)+
            for match in re.finditer('j*n+', pos_tags):
                start = match.start()
                end = match.end()
                word_indices = range(start, end)

                keyphrase = ""
                stemmed_keyphrase_list = []
                unstemmed_keyphrase_list = []
                pos_keyphrase_list = []
                for word_index in word_indices:
                    keyphrase += sentence.stems[word_index] + " "

                    stemmed_keyphrase_list.append(sentence.stems[word_index])
                    unstemmed_keyphrase_list.append(sentence.words[word_index])
                    pos_keyphrase_list.append(sentence.pos[word_index])
                keyphrase = keyphrase[:-1]
                if keyphrase not in candidate_keyphrases:
                    candidate_keyphrases[keyphrase] = dict()
                    candidate_keyphrases[keyphrase]['sentence_id'] = sentence_id
                    candidate_keyphrases[keyphrase]['stemmed'] = stemmed_keyphrase_list
                    candidate_keyphrases[keyphrase]['unstemmed'] = unstemmed_keyphrase_list
                    candidate_keyphrases[keyphrase]['pos'] = pos_keyphrase_list
                    # FIXME: add offset
            # print(sentence.words)
            # print(sentence.meta)
            # print(sentence.stems)
            # print(sentence.pos)
            # print(candidate_keyphrases)

        ################################################################
        # FIXME: select keyphrases that contain 1+ exemplar terms
        ################################################################
        # FIXME: reuse the old dict by deleting unused keyphrases instead?
        filtered_candidate_keyphrases = dict()
        for keyphrase, val in candidate_keyphrases.items():
            # weight = 0
            exemplar_terms_count = 0
            for key, val in cluster_distances.items():
                if tokens[val['centroid_index']] in keyphrase:
                    # FIXME FIXME FIXME FIXME FIXME FIXME
                    # FIXME: weight --> rn I just add the distance but that would mean a smaller weight would be better in that case...
                    # FIXME FIXME FIXME FIXME FIXME FIXME
                    # weight += val['centroid_dist']
                    exemplar_terms_count += 1

            if exemplar_terms_count > 0:
                filtered_candidate_keyphrases[keyphrase] = dict()
                filtered_candidate_keyphrases[keyphrase]['weight'] = 1#weight
                filtered_candidate_keyphrases[keyphrase]['exemplar_terms_count'] = exemplar_terms_count
                filtered_candidate_keyphrases[keyphrase]['sentence_id'] = candidate_keyphrases[keyphrase]['sentence_id']
                filtered_candidate_keyphrases[keyphrase]['stemmed'] = candidate_keyphrases[keyphrase]['stemmed']
                filtered_candidate_keyphrases[keyphrase]['unstemmed'] = candidate_keyphrases[keyphrase]['unstemmed']
                filtered_candidate_keyphrases[keyphrase]['pos'] = candidate_keyphrases[keyphrase]['pos']
                # FIXME: offset
        # print(filtered_candidate_keyphrases)

        ################################################################
        # FIXME: filter out frequent single word candidate keyphrases
        ################################################################

        ##################################
        # FIXME: set the final candidates
        ##################################
        # clean the current candidates list since it is no longer accurate
        self.candidates = defaultdict(Candidate)
        for candidate_keyphrase, vals in filtered_candidate_keyphrases.items():
            offset = 0 # FIXME
            self.add_candidate(vals['unstemmed'], vals['stemmed'], vals['pos'], offset, vals['sentence_id'])

        for k in self.candidates.keys():
            self.weights[k] = 1 # FIXME
