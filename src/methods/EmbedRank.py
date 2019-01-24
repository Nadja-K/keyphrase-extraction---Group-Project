import inspect
import re
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
from common.helper import _create_simple_embedding_visualization


class EmbedRank(LoadFile):
    def __init__(self):
        super(EmbedRank, self).__init__()

    def candidate_selection(self, candidate_selector, **kwargs):
        # Select candidate Keyphrases based on PoS-Tags with a regex
        candidate_selector.select_candidates(self, **kwargs)

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
        tokenized_form_candidate_terms = [candidate.tokenized_form for stemmed_term, candidate in
                                          self.candidates.items()]
        phrase_embeddings = sent2vec_model.get_tokenized_sents_embeddings(tokenized_form_candidate_terms)

        # Filter out candidate phrases that were not found in sent2vec
        valid_candidates_mask = ~np.all(phrase_embeddings == 0, axis=1)
        for term, keep in zip(list(self.candidates), valid_candidates_mask):
            # Note: do NOT write 'if keep is False', somehow this doesn't work???
            if keep == False:
                # print("%s not found in sent2vec, removing candidate" % self.candidates[term].tokenized_form)
                del self.candidates[term]
        phrase_embeddings = phrase_embeddings[valid_candidates_mask, :]

        # Rank the candidate phrases according to their cosine distance to the document embedding
        for index, term in enumerate(self.candidates):
            candidate_embedding = phrase_embeddings[index]

            # compute the cos distance/similarity
            self.weights[term] = cosine_similarity(candidate_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))[0][0]

        # Simple embedding visualization
        # FIXME: enable/disable on parameter
        # _create_simple_embedding_visualization(phrase_embeddings, list(self.candidate_terms), list(range(len(self.candidate_terms))), doc_embedding, 'test')