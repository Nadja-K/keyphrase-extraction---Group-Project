import numpy as np
import scipy as sp
import glob
from common.EmbeddingDistributor import EmbeddingDistributor
from methods.EmbedRank import EmbedRank
from common.CandidateSelector import CandidateSelector, embed_rank_candidate_selector

# change these depending on the language
language = 'en'
regex = 'a*n+'
sent2vec_model_name = '../word_embedding_models/english/sent2vec/wiki_bigrams.bin'

# don't change these
normalization = 'stemming'

# data to use (needs to be adapted for the database later) # FIXME
# train_folder = "../ake-datasets/datasets/SemEval-2010/train"
# train_folder = "../ake-datasets/datasets/Inspec/train"
train_folder = "../ake-datasets/datasets/DUC-2001/test"
output_file = 'DUC-2001_covariance'

centroid_output_file = f'{output_file}_centroid'

sent2vec_model = EmbeddingDistributor(sent2vec_model_name)
candidate_selector = CandidateSelector(embed_rank_candidate_selector)

# Collect all candidates
candidate_strings = set()
for file in glob.glob(train_folder + '/*'):
    extractor = EmbedRank()
    extractor.load_document(file, language=language, normalization=normalization)
    extractor.candidate_selection(candidate_selector=candidate_selector, regex=regex, language=language)
    for term, candidate in extractor.candidates.items():
        candidate_strings.add(candidate.tokenized_form)

print("Candidate selection done - Computing embeddings")

candidate_embeddings = sent2vec_model.get_tokenized_sents_embeddings(candidate_strings)
print("Candidate embeddings computed")

# Filter out candidate phrases that were not found in sent2vec
valid_candidates_mask = ~np.all(candidate_embeddings == 0, axis=1)
candidate_embeddings = candidate_embeddings[valid_candidates_mask, :]
print(candidate_embeddings.shape)

covariance_matrix = sp.cov(candidate_embeddings.T)
inverse_covariance_matrix = sp.linalg.inv(covariance_matrix)
print(inverse_covariance_matrix.shape)

centroid = candidate_embeddings.mean(axis=0)
print(centroid)

np.save(output_file, covariance_matrix)
np.save(centroid_output_file, centroid)