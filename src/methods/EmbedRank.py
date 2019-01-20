import inspect

import spacy
import sent2vec
from pke.base import LoadFile
from pke.data_structures import Candidate
from collections import defaultdict

from common.CandidateSelector import CandidateSelector
from common.DatabaseHandler import DatabaseHandler
from common.KeyphraseSelector import KeyphraseSelector


class EmbedRank(LoadFile):
    def __init__(self):
        super(EmbedRank, self).__init__()
        self.candidate_terms = None
        self.cluster_features = None
        self.sent2vec_model = sent2vec.Sent2vecModel()
        # self.sent2vec_model.load_model(fasttext_model)

    def candidate_selection(self, candidate_selector, **kwargs):
        # Select candidate Keyphrases based on PoS-Tags with a regex
        self.candidate_terms = candidate_selector.select_candidates(self, **kwargs)
        print(list(self.candidates))

    def candidate_weighting(self, cluster_feature_calculator, cluster_method, exemplar_terms_dist_func, keyphrase_selector, filename, regex='a*n+', num_clusters=0, frequent_word_list=[], draw_graphs=False):
        # FIXME
        # Compute the document embedding based on only nouns and adjectives

        # Compute the phrase embedding for each candidate phrase

        # Rank the candidate phrases according to their cosine distance to the document embedding

        pass

    def _collect_data(self, clusters, cluster_exemplar_terms, unfiltered_candidate_keyphrases, candidate_keyphrases):
        # FIXME
        # terms = list(self.candidate_terms)
        # for index, cluster in enumerate(clusters):
        #     term = terms[index]
        #     self.data_cluster_members[term] = {
        #         'surface_forms': self.candidate_terms[term].surface_forms,
        #         'lexical_form': self.candidate_terms[term].lexical_form,
        #         'pos_patterns': self.candidate_terms[term].pos_patterns,
        #         'offsets': self.candidate_terms[term].offsets,
        #         'sentence_ids': self.candidate_terms[term].sentence_ids,
        #         'cluster': int(cluster),
        #         'exemplar_term': term == cluster_exemplar_terms[cluster]['term']
        #     }
        #
        # self.data_candidate_keyphrases = unfiltered_candidate_keyphrases
        # for candidate in self.data_candidate_keyphrases:
        #     self.data_candidate_keyphrases[candidate]['selected'] = candidate in candidate_keyphrases
        pass
