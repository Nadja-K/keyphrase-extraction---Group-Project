import pke
from pke.unsupervised import (
    TopicRank, SingleRank,
    MultipartiteRank, PositionRank,
    TopicalPageRank, ExpandRank,
    TextRank, TfIdf, KPMiner,
    YAKE, FirstPhrases
)
from KeyCluster import KeyCluster
from EmbedRank import EmbedRank

from KeyphraseExtractor import KeyphraseExtractor

from ClusterFeatureCalculator import CooccurrenceClusterFeature, PPMIClusterFeature, WordEmbeddingsClusterFeature
from CandidateSelector import CandidateSelector, embed_rank_candidate_selector
from Cluster import HierarchicalClustering, SpectralClustering
from KeyphraseSelector import KeyphraseSelector
from evaluation import Evaluator, stemmed_wordwise_phrase_compare, stemmed_compare, stemmed_word_compare
from Cluster import euclid_dist

from helper import custom_normalize_POS_tags


pke.base.ISO_to_language['de'] = 'german'
pke.LoadFile.normalize_pos_tags = custom_normalize_POS_tags


kwargs = {
    'language': 'de',
    'normalization': "stemming",
    # 'n_keyphrases': 10,
    # 'redundancy_removal': ,
    # 'n_grams': 1,
    # 'stoplist': ,
    # 'frequency_file': 'data/heise_df_counts.tsv.gz',  # '../ake-datasets/datasets/SemEval-2010/df_counts.tsv.gz',
    # 'window': 2,
    # 'pos': ,
    # 'top_percent': 1.0,
    # 'normalized': ,
    # 'run_candidate_selection': ,
    # 'threshold': ,
    # 'method': 'ward', # COMMENT OUT FOR TopicRank!
    # 'heuristic': ,
    # 'alpha': ,
    # 'grammar': ,
    # 'lda_model': ,
    # 'maximum_word_number': ,
    # 'lasf': ,
    # 'cutoff': ,
    # 'sigma': ,

    # 'candidate_selector': CandidateSelector(key_cluster_candidate_selector),
    'cluster_feature_calculator': WordEmbeddingsClusterFeature,#PPMIClusterFeature,
    # word_embedding_comp_func': sklearn.metrics.pairwise.cosine_similarity,#np.dot,
    # 'global_cooccurrence_matrix': 'heise_out.cooccurrence',#'inspec_out.cooccurrence',#'semeval_out.cooccurrence',
    # 'cluster_method': SpectralClustering,
    # 'keyphrase_selector': ,
    'regex': 'n{1,3}',
    # 'num_clusters': 20,
    # 'cluster_calc': ,
    'factor': 1/10,
    'frequent_word_list_file': 'data/frequent_word_lists/de_50k.txt',
    'min_word_count': 1000,
    # 'frequent_word_list': ['test'],
    'word_embedding_model_file': "/video2/keyphrase_extraction/word_embedding_models/german/devmount/la_vectors_devmount",
    # 'word_embedding_model':
    'evaluator_compare_func': [stemmed_compare, stemmed_wordwise_phrase_compare],

    # 'filter_reference_keyphrases': True # ONLY USE FOR KEYCLUSTER CHECKING!,
    # 'draw_graphs': True,
    # 'print_document_scores': False,

    'num_documents': 200,
    'batch_size': 100,
    'reference_table': 'stemmed_filtered_stemmed',
    # 'table': 'pos_tags',
    'write_to_db': False
}


def heise_eval():
    extractor = KeyphraseExtractor()
    models = [
        KeyCluster,
        # TfIdf,
        # TopicRank,
        # SingleRank,
        # TextRank,
        # KPMiner
    ]

    # print("Computing the document frequency file.")
    # compute_db_document_frequency("test_df_counts.tsv.gz", **kwargs)

    # print("Computing the global cooccurrence matrix.")
    # compute_global_cooccurrence("heise_out.cooccurrence", **kwargs)

    for m in models:
        print("Computing the F-Score for the Heise Dataset with {}".format(m))
        evaluators = extractor.calculate_model_f_score(m, **kwargs)
        print("\n\n")
        for key, evaluator_data in evaluators.items():
            macro_precision = evaluator_data['macro_precision']
            macro_recall = evaluator_data['macro_recall']
            macro_f_score = evaluator_data['macro_f_score']

            print("%s - Macro average precision: %s, recall: %s, f-score: %s" % (
            key, macro_precision, macro_recall, macro_f_score))


def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    heise_eval()


if __name__ == '__main__':
    main()
