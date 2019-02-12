import numpy as np
import glob

from common.DatabaseHandler import DatabaseHandler
from common.EmbeddingDistributor import EmbeddingDistributor
from methods.EmbedRank import EmbedRank
from common.CandidateSelector import CandidateSelector, embed_rank_candidate_selector
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import pandas as pd
from multiprocessing import Pool
from itertools import repeat
import os


def collect_candidates(document_embeddings, candidate_strings, extractor, filename, sent2vec_model, language,
                       candidate_selector, regex):
    # Candidate selection
    extractor.candidate_selection(candidate_selector=candidate_selector, regex=regex, language=language)

    # Collect the tokenized form of the candidates (which is needed for EmbedRank)
    for term, candidate in extractor.candidates.items():
        candidate_strings.add(candidate.tokenized_form)

    # Compute the document embedding as well
    extractor._compute_document_embedding(sent2vec_model, language)
    document_embeddings[filename] = extractor.doc_embedding

    return document_embeddings, candidate_strings


def compute_candidate_embeddings(sent2vec_model, candidate_strings):
    candidate_strings = list(candidate_strings)
    candidate_embeddings = sent2vec_model.get_tokenized_sents_embeddings(candidate_strings)
    print("Candidate embeddings computed")

    # Filter out candidate phrases that were not found in sent2vec
    valid_candidates_mask = ~np.all(candidate_embeddings == 0, axis=1)
    candidate_embeddings = candidate_embeddings[valid_candidates_mask, :]

    # Filter out candidate phrases from the string list as well
    filtered_candidate_strings = []
    for index, keep in enumerate(valid_candidates_mask):
        if keep == True:
            filtered_candidate_strings.append(candidate_strings[index])

    return candidate_embeddings, filtered_candidate_strings


def compute_candidate_document_similarity(candidate, candidate_embedding, document_embeddings):
    similarity_data = dict()
    similarity_data[candidate] = dict()

    for document, doc_embedding in document_embeddings.items():
        similarity_data[candidate][document] = \
        cosine_similarity(candidate_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))[0][0]

    return similarity_data


def compute_all_candidate_document_similarity(candidate_strings, candidate_embeddings, document_embeddings):
    # Compute the candidate - document similarity
    similarity_data = dict()

    input_data = zip(candidate_strings, candidate_embeddings, repeat(document_embeddings))
    try:
        pool = Pool(os.cpu_count()-1)
        results = pool.starmap(compute_candidate_document_similarity, input_data)
    finally:
        pool.close()
        pool.join()
        [similarity_data.update(res) for res in results]

    # for candidate, candidate_embedding in zip(candidate_strings, candidate_embeddings):
    #     if candidate not in similarity_data:
    #         similarity_data[candidate] = dict()
    #
    #     for document, doc_embedding in document_embeddings.items():
    #         similarity_data[candidate][document] = cosine_similarity(candidate_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))[0][0]

        # break;
    similarity_dataframe = pd.DataFrame(similarity_data)
    similarity_dataframe = similarity_dataframe.reindex(sorted(similarity_dataframe.columns), axis=1)
    ranking_dataframe = similarity_dataframe.rank(axis='index', ascending=False)

    return similarity_dataframe, ranking_dataframe


def main():
    # change these depending on the language
    language = 'de'#'en'
    regex = 'n{1,3}'#'a*n+'
    sent2vec_model_name = '../word_embedding_models/german/sent2vec/de_model.bin'#'../word_embedding_models/english/sent2vec/wiki_bigrams.bin'

    # don't change these
    normalization = 'stemming'

    # data to use
    dataset = 'Heise'  # SemEval-2010, Inspec, DUC-2001, Heise
    if dataset == 'SemEval-2010':
        input_data = "../ake-datasets/datasets/SemEval-2010/train"
    elif dataset == 'Inspec':
        input_data = "../ake-datasets/datasets/Inspec/train"
    elif dataset == 'DUC-2001':
        input_data = "../ake-datasets/datasets/DUC-2001/test"
    else:
        input_data = None

    sent2vec_model = EmbeddingDistributor(sent2vec_model_name)
    candidate_selector = CandidateSelector(embed_rank_candidate_selector)

    document_embeddings = dict()
    candidate_strings = set()
    # Candidate Selection based on the dataset
    if dataset == 'Heise':
        db_handler = DatabaseHandler()
        documents, _ = db_handler.load_split_from_db(EmbedRank, dataset=dataset, split='train', reference_table='stemmed_filtered_stemmed')
        for key, doc in documents.items():
            extractor = doc['document']

            document_embeddings, candidate_strings = collect_candidates(document_embeddings, candidate_strings, extractor, key, sent2vec_model, language, candidate_selector, regex)

    elif input_data is not None:
        # Collect all candidates and compute the document embeddings
        index = 0
        for file in sorted(glob.glob(input_data + '/*')):
            extractor = EmbedRank()
            extractor.load_document(file, language=language, normalization=normalization)

            filename = Path(file).name.split('.')[0]
            document_embeddings, candidate_strings = collect_candidates(document_embeddings, candidate_strings, extractor, filename, sent2vec_model, language, candidate_selector, regex)

    else:
        raise Exception("Something went wrong, make sure your input data is declared properly.")

    print("Candidate selection done - Computing embeddings")

    candidate_embeddings, candidate_strings = compute_candidate_embeddings(sent2vec_model, candidate_strings)
    similarity_dataframe, ranking_dataframe = compute_all_candidate_document_similarity(candidate_strings, candidate_embeddings, document_embeddings)

    print(similarity_dataframe)
    # print(ranking_dataframe)

    similarity_dataframe.to_pickle(dataset + '_similarity_dataframe')
    ranking_dataframe.to_pickle(dataset + '_ranking_dataframe')


if __name__ == '__main__':
    main()
