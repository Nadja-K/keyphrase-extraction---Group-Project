import pke
from common.KeyphraseExtractor import KeyphraseExtractor
from common.helper import custom_normalize_POS_tags
from methods.KeyCluster import KeyCluster
from methods.EmbedRank import EmbedRank


pke.base.ISO_to_language['de'] = 'german'
pke.LoadFile.normalize_pos_tags = custom_normalize_POS_tags


kwargs = {
    'language': 'de',
    'normalization': "stemming",
    # 'n_keyphrases': 10,

    ## KeyCluster
    # 'candidate_selector': CandidateSelector(key_cluster_candidate_selector),
    # 'cluster_feature_calculator': WordEmbeddingsClusterFeature,
    # 'word_embedding_comp_func': sklearn.metrics.pairwise.cosine_similarity,
    # 'global_cooccurrence_matrix': 'data/global_cooccurrence/heise_train.cooccurrence',
    # 'global_cooccurrence_constant': 0,
    # 'regex': 'n{1,3}',
    # 'factor': 1/10,
    'frequent_word_list_file': 'data/frequent_word_lists/de_50k.txt',
    'min_word_count': 1000,
    # 'word_embedding_model_file': "/video2/keyphrase_extraction/word_embedding_models/german/devmount/la_vectors_devmount",

    ## EmbedRank
    # 'sent2vec_model': '../word_embedding_models/german/sent2vec/de_model.bin',
    # 'document_similarity': False,
    # 'document_similarity_new_candidate_constant': 1.0,
    # 'document_similarity_weights': (1.0, 1.0),
    # 'global_covariance': False,
    # 'global_covariance_weights': (4.0, 0.1),

    'print_document_scores': False,
    'write_to_db': False
}


def extract_keyphrases_from_raw_text():
    extractor = KeyphraseExtractor()
    models = [
        KeyCluster,
        # EmbedRank,
    ]

    for m in models:
        keyphrases, unstemmed_keyphrases = extractor.extract_keyphrases_from_raw_text(m, 'test_input.txt', **kwargs)

        print(keyphrases)
        # Additionally print the unstemmed keyphrases if stemming was used
        if kwargs.get('normalization', 'stemming') == 'stemming':
            print(unstemmed_keyphrases)


def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    extract_keyphrases_from_raw_text()


if __name__ == '__main__':
    main()
