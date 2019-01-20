import string
from collections import defaultdict
from pke.data_structures import Candidate
from typing import Callable
from KeyphraseSelector import _MAPPING

# Switch the key and values from _MAPPING
_MIRRORED_MAPPING = y_dict2 = {y:x for x,y in _MAPPING.items()}


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


def embed_rank_candidate_selector(context, **kwargs):
    regex = kwargs.get('regex', 'a*n+')

    # translate the regex to the correct grammar, only works for simple grammars right now
    translated_regex = "NP:{"
    for char in regex:
        mapped_char = _MIRRORED_MAPPING.get(char, None)
        if mapped_char is not None:
            translated_regex += "<" + mapped_char + ">"
        else:
            translated_regex += char
    translated_regex += "}"
    print("Using the following regex for candidate selection: %s" % translated_regex)

    # select unigrams as possible candidates
    context.candidates = defaultdict(Candidate)
    context.grammar_selection(grammar=translated_regex)

    return context.candidates.copy()


class CandidateSelector:
    def __init__(self, select_func: Callable = key_cluster_candidate_selector):
        self.select_func = select_func

    def select_candidates(self, context, **kwargs):
        return self.select_func(context, **kwargs)
