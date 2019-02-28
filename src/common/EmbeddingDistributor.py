# Based on EmbedRank, Swisscom (Schweiz) AG.
# Original Authors: Kamil Bennani-Smires, Yann Savary

import sent2vec

class EmbeddingDistributor():
    def __init__(self, fasttext_model):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(fasttext_model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        Generate a numpy ndarray with the embedding of each element of sent in each row

        :param sents: list of string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        return self.model.embed_sentences(sents)
