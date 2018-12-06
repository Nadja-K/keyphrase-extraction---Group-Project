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
                                 mininum_length=1, mininum_word_size=1, only_alphanum=False)
        cooccurrence_terms = list(context.candidates.copy())

        # Calculate cooccurrence_matrix
        cooccurrence_matrix = np.zeros((len(filtered_candidate_terms), len(filtered_candidate_terms)))
        for sentence in list(context.sentences):
            words = sentence.stems.copy()

            # Remove words/symbols that don't appear in the punctuation filtered tokens list
            for index in sorted([i for i, x in enumerate(words) if x not in cooccurrence_terms], reverse=True):
                words.pop(index)

            #
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

    def calc_cluster_features(self, context, candidate_terms):
        ppmi_matrix = []
        cooccurrence_matrix = super().calc_cluster_features(context, candidate_terms)

        # FIXME

        return ppmi_matrix
