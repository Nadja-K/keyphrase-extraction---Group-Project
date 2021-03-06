import string
import numpy as np
from collections import defaultdict
import sys

import sklearn
import spacy
from pke.data_structures import Candidate
from sklearn.metrics.pairwise import cosine_similarity


class CooccurrenceClusterFeature:
    def __init__(self, **kwargs):
        self.window = kwargs.get('window', 2)
        self.global_cooccurrence_matrix = kwargs.get('global_cooccurrence_matrix', None)
        self.global_cooccurrence_constant = kwargs.get('global_cooccurrence_constant', 0)

    def _reduce_global_cooccurrence_matrix(self, filtered_candidate_terms):
        # print("Using global cooccurrence matrix")
        keys = self.global_cooccurrence_matrix['keys']
        full_matrix = self.global_cooccurrence_matrix['cooccurrence_matrix']

        if self.global_cooccurrence_constant == 'mean':
            self.global_cooccurrence_constant = full_matrix.mean()
            print(self.global_cooccurrence_constant)

        reduced_cooccurrence_matrix = np.zeros((len(filtered_candidate_terms), len(filtered_candidate_terms)))
        for local_word_index1, word1 in enumerate(filtered_candidate_terms):
                global_word_index1 = keys.index(word1) if word1 in keys else -1
                for local_word_index2, word2 in enumerate(filtered_candidate_terms):
                    if local_word_index2 < local_word_index1:
                        continue
                    else:
                        global_word_index2 = keys.index(word2) if word2 in keys else -1

                        # Handle words that do not appear in the full global cooccurrence matrix
                        if global_word_index1 == -1 or global_word_index2 == -1:
                            reduced_cooccurrence_matrix[local_word_index1][local_word_index2] = self.global_cooccurrence_constant
                            reduced_cooccurrence_matrix[local_word_index2][local_word_index1] = self.global_cooccurrence_constant
                        else:
                            reduced_cooccurrence_matrix[local_word_index1][local_word_index2] = full_matrix[global_word_index1][global_word_index2]
                            reduced_cooccurrence_matrix[local_word_index2][local_word_index1] = reduced_cooccurrence_matrix[local_word_index1][local_word_index2]

        return reduced_cooccurrence_matrix

    def calc_cluster_features(self, context, filtered_candidate_terms):
        filtered_candidate_terms = list(filtered_candidate_terms)
        if self.global_cooccurrence_matrix is not None:
            cooccurrence_matrix = self._reduce_global_cooccurrence_matrix(filtered_candidate_terms)
        else:
            # Get cooccurrence terms including stopwords but filter out punctuation and linebreaks
            context.candidates = defaultdict(Candidate)
            context.ngram_selection(n=1)
            context.candidate_filtering(stoplist=list(string.punctuation) +
                                              ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'],
                                     minimum_length=1, minimum_word_size=1, only_alphanum=False)
            cooccurrence_terms = list(context.candidates.copy())

            # Calculate cooccurrence_matrix
            cooccurrence_matrix = np.zeros((len(filtered_candidate_terms), len(filtered_candidate_terms)))
            for sentence in list(context.sentences):
                words = sentence.stems.copy()

                # Remove words/symbols that don't appear in the punctuation filtered tokens list
                for index in sorted([i for i, x in enumerate(words) if x not in cooccurrence_terms], reverse=True):
                    words.pop(index)

                # Calculate the cooccurrence within a set window size
                for pos in range(len(words)):
                    start = pos - self.window
                    end = pos + self.window + 1

                    # Skip stopwords
                    if words[pos] not in filtered_candidate_terms:
                        continue

                    word_index = filtered_candidate_terms.index(words[pos])

                    if start < 0:
                        start = 0

                    for word in words[start:pos] + words[pos + 1:end]:
                        # Skip stopwords
                        if word in filtered_candidate_terms:
                            cooccurrence_matrix[word_index][filtered_candidate_terms.index(word)] += 1

        return cooccurrence_matrix


class PPMIClusterFeature(CooccurrenceClusterFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_cluster_features(self, context, filtered_candidate_terms):
        filtered_candidate_terms = list(filtered_candidate_terms)
        assert self.global_cooccurrence_matrix is not None, "PPMI needs a global cooccurrence matrix, please specify one under 'global_cooccurrence_matrix."

        cooccurrence_matrix = super()._reduce_global_cooccurrence_matrix(filtered_candidate_terms)
        word_counts = self.global_cooccurrence_matrix['word_counts']
        num_words_total = self.global_cooccurrence_matrix['num_words_total']

        ppmi_matrix = np.zeros((len(filtered_candidate_terms), len(filtered_candidate_terms)))
        for index1, word1 in enumerate(filtered_candidate_terms):
            for index2, word2 in enumerate(filtered_candidate_terms[index1:]):
                index2 = index2 + index1
                # if a word does not appear in the global co occurrence matrix we set the ppmi value to a constant
                if word1 not in word_counts or word2 not in word_counts:
                    # ppmi = 0
                    cooccurrence_count = cooccurrence_matrix[index1][index2]
                    ppmi = max(np.log2((cooccurrence_count * num_words_total) / (sys.float_info.epsilon)), 0)
                else:
                    word1_count = word_counts[word1]
                    word2_count = word_counts[word2]
                    cooccurrence_count = cooccurrence_matrix[index1][index2]

                    ppmi = max(np.log2((cooccurrence_count * num_words_total) / (word1_count * word2_count)), 0)

                ppmi_matrix[index1][index2] = ppmi
        ppmi_matrix = ppmi_matrix + np.triu(ppmi_matrix, k=1).T
        return ppmi_matrix


class WordEmbeddingsClusterFeature:
    def __init__(self, **kwargs):
        self.nlp = kwargs.get('word_embedding_model', None)
        if self.nlp is None:
            print("Loading word embedding model in ClusterFeatureCalculator")
            self.nlp = spacy.load('en_vectors_web_lg')
        self.comp_func = kwargs.get('word_embedding_comp_func', sklearn.metrics.pairwise.cosine_similarity)

    def calc_cluster_features(self, context, filtered_candidate_terms):
        word_embedding_matrix = np.zeros((len(filtered_candidate_terms), len(filtered_candidate_terms)))

        word_embeddings = dict()
        for key, candidate in filtered_candidate_terms.items():
            # make a set of all surface forms in lower case
            surface_forms = set([item.lower() for sublist in candidate.surface_forms for item in sublist])

            # save all surface forms that have a word embedding vector
            surface_forms_in_model = ""
            for surface_form in surface_forms:
                token = self.nlp(surface_form)
                if token.has_vector is True:
                    surface_forms_in_model += token.text + " "

            # Remove the possible excess space and get the (mean) embedding vector
            word_embeddings[candidate] = self.nlp(surface_forms_in_model[:-1]).vector.reshape(1, -1)

        # Calculate the similarity matrix
        for index1, candidate1 in enumerate(word_embeddings):
            for index2, candidate2 in enumerate(word_embeddings):
                if index2 < index1:
                    continue

                # Different similarity metrics need to be handled differently
                if hasattr(sklearn.metrics.pairwise, self.comp_func.__name__):
                    if self.comp_func is sklearn.metrics.pairwise.cosine_similarity:
                        word_embedding_matrix[index1][index2] = min(
                            max(self.comp_func(word_embeddings[candidate1], word_embeddings[candidate2])[0][0], -1), 1)
                    else:
                        word_embedding_matrix[index1][index2] = \
                        self.comp_func(word_embeddings[candidate1], word_embeddings[candidate2])[0][0]
                else:
                    word_embedding_matrix[index1][index2] = self.comp_func(word_embeddings[candidate1][0], word_embeddings[candidate2][0])

                word_embedding_matrix[index2][index1] = word_embedding_matrix[index1][index2]

        return word_embedding_matrix
