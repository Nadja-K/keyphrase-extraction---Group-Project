from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import numpy as np
from pke.base import LoadFile
from common.helper import _create_simple_embedding_visualization


class EmbedRank(LoadFile):
    def __init__(self):
        super(EmbedRank, self).__init__()

    def candidate_selection(self, candidate_selector, **kwargs):
        # Select candidate Keyphrases based on PoS-Tags with a regex
        candidate_selector.select_candidates(self, **kwargs)

    def candidate_weighting(self, sent2vec_model, filename, draw_graphs=False, language='en'):
        # Compute the document embedding based on only nouns and adjectives
        self._compute_document_embedding(sent2vec_model, language)

        # Compute the phrase embedding for each candidate phrase
        self._compute_phrase_embeddings(sent2vec_model)

        # Rank the candidate phrases according to their cosine distance to the document embedding
        self._rank_candidates()

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
            # Note: do NOT write 'if keep is False', somehow this doesn't work???
            if keep == False:
                # print("%s not found in sent2vec, removing candidate" % self.candidates[term].tokenized_form)
                del self.candidates[term]
        self.phrase_embeddings = self.phrase_embeddings[valid_candidates_mask, :]

    def _rank_candidates(self):
        for index, candidate_tuple in enumerate(sorted(self.candidates.items())):
            term, candidate = candidate_tuple
            candidate_embedding = self.phrase_embeddings[index]

            # compute the cos distance/similarity
            self.weights[term] = cosine_similarity(candidate_embedding.reshape(1, -1), self.doc_embedding.reshape(1, -1))[0][0]