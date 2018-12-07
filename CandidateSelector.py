import string
from collections import defaultdict
from pke.data_structures import Candidate
from typing import Callable


def key_cluster_candidate_selector(context, **kwargs):
    n_grams = kwargs.get('n_grams', 1)
    stoplist = kwargs.get('stoplist', None)

    # select unigrams as possible candidates
    context.candidates = defaultdict(Candidate)
    context.ngram_selection(n=n_grams)

    # remove stop words
    if stoplist is None:
        stoplist = context.stoplist

    context.candidate_filtering(stoplist=list(string.punctuation) +
                                         ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] + stoplist)
    filtered_candidate_terms = context.candidates.copy()
    return filtered_candidate_terms


class CandidateSelector:
    def __init__(self, select_func: Callable = key_cluster_candidate_selector):
        self.select_func = select_func

    def select_candidates(self, context, **kwargs):
        return self.select_func(context, **kwargs)
