import inspect
from itertools import chain

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import numpy as np
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

    def candidate_selection(self, candidate_selector, **kwargs):
        # Select candidate Keyphrases based on PoS-Tags with a regex
        self.candidate_terms = candidate_selector.select_candidates(self, **kwargs)

    def candidate_weighting(self, sent2vec_model):
        # FIXME methoden auslagern

        # Compute the document embedding based on only nouns and adjectives
        tokenized_doc_text = ""
        for sent in self.sentences:
            indices = np.where(np.array(sent.pos) == 'NOUN')[0]
            indices = sorted(np.append(indices, np.where(np.array(sent.pos) == 'ADJ')[0]))

            for index in indices:
                # Note: EmbedRank makes a lower here for the document but not later for the candidates
                # FIXME: find out if lower is better or if the whole method works better without the lower part
                tokenized_doc_text = tokenized_doc_text + " " + sent.words[index]#.lower()
        tokenized_doc_text = tokenized_doc_text[1:]
        doc_embedding = sent2vec_model.get_tokenized_sents_embeddings([tokenized_doc_text])


        # Compute the phrase embedding for each candidate phrase
        tokenized_form_candidate_terms = [' '.join(candidate.tokenized_form) for stemmed_term, candidate in self.candidate_terms.items()]
        phrase_embeddings = sent2vec_model.get_tokenized_sents_embeddings(tokenized_form_candidate_terms)

        # Filter out candidate phrases that were not found in sent2vec
        valid_candidates_mask = ~np.all(phrase_embeddings == 0, axis=1)
        for term, keep in zip(list(self.candidate_terms), valid_candidates_mask):
            # Note: do NOT write 'if keep is False', somehow this doesn't work???
            if keep == False:
                print("%s not found in sent2vec, removing candidate" % term)
                del self.candidate_terms[term]
        phrase_embeddings = phrase_embeddings[valid_candidates_mask, :]


        # Rank the candidate phrases according to their cosine distance to the document embedding
        for index, term in enumerate(self.candidate_terms):
            candidate_embedding = phrase_embeddings[index]

            # compute the cos distance/similarity
            # self.weights[term] = cosine_distances(candidate_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))
            self.weights[term] = cosine_similarity(candidate_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))
            # print(self.weights[term])

        # Collect data and write it to the database
        # FIXME

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
