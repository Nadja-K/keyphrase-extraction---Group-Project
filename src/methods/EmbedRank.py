from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.spatial.distance import cdist

import numpy as np
from pke.base import LoadFile
from common.helper import _create_simple_embedding_visualization

dist_list = []

class EmbedRank(LoadFile):
    def __init__(self):
        super(EmbedRank, self).__init__()

    def candidate_selection(self, candidate_selector, **kwargs):
        # Select candidate Keyphrases based on PoS-Tags with a regex
        candidate_selector.select_candidates(self, **kwargs)

    def candidate_weighting(self, sent2vec_model, filename, draw_graphs=False, language='en', document_similarity_data=None, document_similarity_new_candidate_constant=1.0, document_similarity_weights=None, global_covariance_matrix=None, global_embedding_centroid=None, global_covariance_weights=None):
        # Compute the document embedding based on only nouns and adjectives
        self._compute_document_embedding(sent2vec_model, language)

        # Compute the phrase embedding for each candidate phrase
        self._compute_phrase_embeddings(sent2vec_model)

        # Rank the candidate phrases according to their cosine distance to the document embedding
        self._rank_candidates(document_similarity_data, document_similarity_new_candidate_constant, document_similarity_weights, global_covariance_matrix, global_embedding_centroid, global_covariance_weights)

        # Simple embedding visualization
        if draw_graphs is True:
            _create_simple_embedding_visualization(self.phrase_embeddings, list(self.candidate_terms), list(range(len(self.candidate_terms))), self.doc_embedding, filename)

    def _compute_document_embedding(self, sent2vec_model, language):
        tokenized_doc_text = ""
        for sent in self.sentences:
            indices = np.where(np.array(sent.pos) == 'NOUN')[0]
            indices = sorted(np.append(indices, np.where(np.array(sent.pos) == 'ADJ')[0]))

            for index in indices:
                # Note: EmbedRank makes a lower here for the document but not later for the candidates
                if language == 'en':
                    tokenized_doc_text = tokenized_doc_text + " " + sent.words[index].lower()
                else:
                    tokenized_doc_text = tokenized_doc_text + " " + sent.words[index]
        tokenized_doc_text = tokenized_doc_text[1:]
        self.doc_embedding = sent2vec_model.get_tokenized_sents_embeddings([tokenized_doc_text])

    def _compute_phrase_embeddings(self, sent2vec_model):
        sorted_candidates = sorted(self.candidates.items())
        tokenized_form_candidate_terms = [candidate.tokenized_form for stemmed_term, candidate in
                                          sorted_candidates]
        self.phrase_embeddings = sent2vec_model.get_tokenized_sents_embeddings(tokenized_form_candidate_terms)

        # Filter out candidate phrases that were not found in sent2vec
        valid_candidates_mask = ~np.all(self.phrase_embeddings == 0, axis=1)
        for candidate_tuple, keep in zip(sorted_candidates, valid_candidates_mask):
            term, candidate = candidate_tuple
            if keep == False:
                del self.candidates[term]
        self.phrase_embeddings = self.phrase_embeddings[valid_candidates_mask, :]

    def _rank_candidates(self, document_similarity_data=None, document_similarity_new_candidate_constant=1.0, document_similarity_weights=None, global_covariance_matrix=None, global_embedding_centroid=None, global_covariance_weights=None):
        for index, candidate_tuple in enumerate(sorted(self.candidates.items())):
            term, candidate = candidate_tuple
            candidate_embedding = self.phrase_embeddings[index]

            # compute the cos distance/similarity
            self.weights[term] = (cosine_similarity(candidate_embedding.reshape(1, -1), self.doc_embedding.reshape(1, -1))[0][0])# + candidate_document_similarity_bias

            # Add the document similarity bias based on the ranking of the term
            # Candidates that are not found in the document similarity matrix are treated as special and receive a
            # bias value of 1 [This might be changed and could be a hyperparameter] (very important for this document
            # since they did not appear in any of the training documents)
            if document_similarity_data is not None:
                assert type(document_similarity_weights) is tuple, "document_similarity_weights is not a tuple: %r" % document_similarity_weights

                if candidate.tokenized_form in document_similarity_data.columns:
                    candidate_column = document_similarity_data[candidate.tokenized_form]
                    candidate_column = candidate_column.append(DataFrame({0: {'Current Document': self.weights[term]}}))
                    candidate_column = candidate_column.rank(axis='index', ascending=False)
                    candidate_document_similarity_bias = (candidate_column.at['Current Document', 0] / candidate_column.max())[0]
                else:
                    candidate_document_similarity_bias = document_similarity_new_candidate_constant

                self.weights[term] = self.weights[term] * document_similarity_weights[0] + candidate_document_similarity_bias * document_similarity_weights[1]

            if global_covariance_matrix is not None:
                mahalanobis_dist = cdist(candidate_embedding.reshape(1, -1), global_embedding_centroid, 'mahalanobis', VI=global_covariance_matrix)[0][0]
                norm_mahalanobis_dist = mahalanobis_dist/31.443647329673244

                self.weights[term] = self.weights[term] * global_covariance_weights[0] + (1-norm_mahalanobis_dist) * global_covariance_weights[1]
