import string
from collections import defaultdict
from pke.data_structures import Candidate


class CandidateTermSelector:
    def select_candidates(self, context, ngrams=1, stoplist=None):
        # select unigrams as possible candidates
        context.candidates = defaultdict(Candidate)
        context.ngram_selection(n=ngrams)

        # remove stop words
        if stoplist is None:
            stoplist = context.stoplist

        context.candidate_filtering(stoplist=list(string.punctuation) +
                                             ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] + stoplist)
        filtered_candidate_terms = context.candidates.copy()
        return filtered_candidate_terms
