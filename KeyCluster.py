from pke.base import LoadFile
from pke.data_structures import Candidate
from collections import defaultdict

class KeyCluster(LoadFile):
    def __init__(self):
        super(KeyCluster, self).__init__()
        self.candidate_terms = None
        self.cluster_features = None

    def candidate_selection(self, candidate_selector, **kwargs):
        self.candidate_terms = list(candidate_selector.select_candidates(self, **kwargs))

    def candidate_weighting(self, cluster_feature_calculator, cluster_method, exemplar_terms_dist_func, keyphrase_selector, regex='a*n+', num_clusters=0, frequent_word_list=[]):
        # Calculating term relatedness
        self.cluster_features = cluster_feature_calculator.calc_cluster_features(self, self.candidate_terms)

        # Term clustering
        if num_clusters == 0:
            num_clusters = int(2./3. * len(self.candidate_terms))
        clusters = cluster_method.calc_clusters(num_clusters, self.cluster_features)
        cluster_exemplar_terms = cluster_method.get_exemplar_terms(clusters, self.cluster_features, exemplar_terms_dist_func)

        # Create candidate keyphrases
        candidate_keyphrases = keyphrase_selector.select_candidate_keyphrases(self.sentences, regex=regex)

        # Select keyphrases that contain 1+ exemplar terms
        candidate_keyphrases = keyphrase_selector.filter_candidate_keyphrases(candidate_keyphrases,
                                                                              self.candidate_terms,
                                                                              cluster_exemplar_terms)


        # filter out frequent single word candidate keyphrases
        # based on subtitles (all languages): https://github.com/hermitdave/FrequencyWords/
        # based on wikipedia (english only): https://github.com/IlyaSemenov/wikipedia-word-frequency
        # print("frequent word list: %s" % frequent_word_list)
        candidate_keyphrases = keyphrase_selector.frequent_word_filtering(frequent_word_list, candidate_keyphrases)

        # Set the final candidates
        # clean the current candidates list since it is no longer accurate
        self.candidates = defaultdict(Candidate)
        for candidate_keyphrase, vals in candidate_keyphrases.items():
            self.add_candidate(vals['unstemmed'], vals['stemmed'], vals['pos'], vals['char_offsets'], vals['sentence_id'])
            self.weights[candidate_keyphrase] = vals['weight']
