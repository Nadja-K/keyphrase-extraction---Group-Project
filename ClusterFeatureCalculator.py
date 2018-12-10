import string
import numpy as np
from collections import defaultdict

from pke.data_structures import Candidate


class CooccurrenceClusterFeature:
    def __init__(self, window=2):
        self.window = window

    def calc_cluster_features(self, context, filtered_candidate_terms):
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
    def __init__(self, window=2):
        super().__init__(window=window)

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
