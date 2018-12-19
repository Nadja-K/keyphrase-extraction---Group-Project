import string
import numpy as np
from collections import defaultdict

import sklearn
import spacy
from pke.data_structures import Candidate
from sklearn.metrics.pairwise import cosine_similarity

class CooccurrenceClusterFeature:
    def __init__(self, **kwargs):
        self.window = kwargs.get('window', 2)

    def calc_cluster_features(self, context, filtered_candidate_terms):
        filtered_candidate_terms = list(filtered_candidate_terms)
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
        cooccurrence_matrix = super().calc_cluster_features(context, filtered_candidate_terms)

        ppmi_matrix = np.zeros((len(filtered_candidate_terms), len(filtered_candidate_terms)))
        for index1, word1 in enumerate(filtered_candidate_terms):
            word1_count = len(context.candidates[word1].offsets)
            for index2, word2 in enumerate(filtered_candidate_terms[index1:]):
                index2 = index2 + index1
                word2_count = len(context.candidates[word2].offsets)
                cooccurrence_count = cooccurrence_matrix[index1][index2]
                ppmi = max(np.log2(cooccurrence_count / (word1_count * word2_count)), 0)
                # print(np.log2(cooccurrence_count / (word1_count * word2_count)), cooccurrence_count, word1_count, word2_count)
                ppmi_matrix[index1][index2] = ppmi

        ppmi_matrix = ppmi_matrix + np.triu(ppmi_matrix, k=1).T
        # print(ppmi_matrix)

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
