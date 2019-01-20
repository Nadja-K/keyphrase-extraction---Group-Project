import inspect

import spacy
from pke.base import LoadFile
from pke.data_structures import Candidate
from collections import defaultdict

from common.CandidateSelector import CandidateSelector
from common.DatabaseHandler import DatabaseHandler
from common.KeyphraseSelector import KeyphraseSelector


class KeyCluster(LoadFile):
    def __init__(self):
        super(KeyCluster, self).__init__()
        self.candidate_terms = None
        self.cluster_features = None

        self.data_cluster_members = dict()
        self.data_candidate_keyphrases = dict()

    def candidate_selection(self, candidate_selector, **kwargs):
        self.candidate_terms = candidate_selector.select_candidates(self, **kwargs)

    def candidate_weighting(self, cluster_feature_calculator, cluster_method, exemplar_terms_dist_func, keyphrase_selector, filename, regex='a*n+', num_clusters=0, frequent_word_list=[], draw_graphs=False):
        # Calculating term relatedness
        self.cluster_features = cluster_feature_calculator.calc_cluster_features(self, self.candidate_terms)

        # Term clustering
        if num_clusters == 0:
            num_clusters = int(2./3. * len(list(self.candidate_terms)))
        clusters = cluster_method.calc_clusters(num_clusters, self.cluster_features, self.candidate_terms, filename=filename, draw_graphs=draw_graphs)
        cluster_exemplar_terms = cluster_method.get_exemplar_terms(clusters, self.cluster_features, self.candidate_terms, exemplar_terms_dist_func)

        # Alternative to clustering: pick random exemplar terms
        # cluster_exemplar_terms, clusters = cluster_method.get_random_exemplar_terms(num_clusters, self.candidate_terms)

        # Create candidate keyphrases
        candidate_keyphrases = keyphrase_selector.select_candidate_keyphrases(self.sentences, regex=regex)

        # Determine how many exemplar terms each candidate keyphrase contains
        candidate_keyphrases, unfiltered_candidate_keyphrases = keyphrase_selector.filter_candidate_keyphrases(candidate_keyphrases,
                                                                              self.candidate_terms,
                                                                              cluster_exemplar_terms)
        # filter out frequent single word candidate keyphrases
        # based on subtitles (all languages): https://github.com/hermitdave/FrequencyWords/
        # based on wikipedia (english only): https://github.com/IlyaSemenov/wikipedia-word-frequency
        candidate_keyphrases = keyphrase_selector.frequent_word_filtering(frequent_word_list, candidate_keyphrases)

        # Set the final candidates
        # clean the current candidates list since it is no longer accurate
        self.candidates = defaultdict(Candidate)
        for candidate_keyphrase, vals in candidate_keyphrases.items():
            # Only keep candidates that contain 1+ exemplar terms
            if vals['exemplar_terms_count'] > 0:
                self.add_candidate(vals['words'], vals['stems'], vals['pos'], vals['char_offsets'], vals['sentence_id'])
                self.weights[candidate_keyphrase] = vals['weight']

        # Collect data that will be written into the database
        self._collect_data(clusters, cluster_exemplar_terms, unfiltered_candidate_keyphrases, candidate_keyphrases)
        return num_clusters

    def _collect_data(self, clusters, cluster_exemplar_terms, unfiltered_candidate_keyphrases, candidate_keyphrases):
        terms = list(self.candidate_terms)
        for index, cluster in enumerate(clusters):
            term = terms[index]
            self.data_cluster_members[term] = {
                'surface_forms': self.candidate_terms[term].surface_forms,
                'lexical_form': self.candidate_terms[term].lexical_form,
                'pos_patterns': self.candidate_terms[term].pos_patterns,
                'offsets': self.candidate_terms[term].offsets,
                'sentence_ids': self.candidate_terms[term].sentence_ids,
                'cluster': int(cluster),
                'exemplar_term': term == cluster_exemplar_terms[cluster]['term']
            }

        self.data_candidate_keyphrases = unfiltered_candidate_keyphrases
        for candidate in self.data_candidate_keyphrases:
            self.data_candidate_keyphrases[candidate]['selected'] = candidate in candidate_keyphrases
