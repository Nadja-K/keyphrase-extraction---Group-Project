import pke
from pke.unsupervised import TfIdf, TextRank, SingleRank, TopicRank, MultipartiteRank

from common.CandidateSelector import key_cluster_candidate_selector, CandidateSelector
from common.KeyphraseExtractor import KeyphraseExtractor
from common.ClusterFeatureCalculator import WordEmbeddingsClusterFeature, PPMIClusterFeature
from common.helper import custom_normalize_POS_tags, compute_global_cooccurrence, compute_db_document_frequency
from methods.EmbedRank import EmbedRank
from methods.KeyCluster import KeyCluster
from eval.evaluation import stemmed_wordwise_phrase_compare, stemmed_compare, word_compare, wordwise_phrase_compare

pke.base.ISO_to_language['de'] = 'german'
pke.LoadFile.normalize_pos_tags = custom_normalize_POS_tags

kwargs = {
    'language': 'de',
    'normalization': "stemming",
    'n_keyphrases': 10,
    # 'redundancy_removal': ,
    # 'n_grams': 1,
    # 'stoplist': ,
    # 'frequency_file': 'data/document_frequency/heise_train_df_counts.tsv.gz',
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

    ## KeyCluster
    # 'candidate_selector': CandidateSelector(key_cluster_candidate_selector),
    # 'cluster_feature_calculator': PPMIClusterFeature,#WordEmbeddingsClusterFeature,#PPMIClusterFeature,
    # 'word_embedding_comp_func': sklearn.metrics.pairwise.cosine_similarity,#sklearn.metrics.pairwise.manhattan_distances, #sklearn.metrics.pairwise.euclidean_distances,#np.dot,#
    # 'global_cooccurrence_matrix': 'data/global_cooccurrence/heise_train.cooccurrence',#'inspec_out.cooccurrence',#'semeval_out.cooccurrence',
    # 'global_cooccurrence_constant': 10,
    # 'cluster_method': SpectralClustering,
    # 'keyphrase_selector': ,
    # 'regex': 'n{1,3}',
    # 'num_clusters': 20,
    # 'cluster_calc': ,
    # 'factor': 1/10,
    # 'frequent_word_list_file': 'data/frequent_word_lists/de_50k.txt',
    # 'min_word_count': 1000,
    # 'frequent_word_list': ['test'],
    # 'word_embedding_model_file': "/video2/keyphrase_extraction/word_embedding_models/german/devmount/la_vectors_devmount",
    # 'word_embedding_model':
    'evaluator_compare_func': [stemmed_compare, stemmed_wordwise_phrase_compare],#[stemmed_compare, stemmed_wordwise_phrase_compare],

    ## EmbedRank
    'sent2vec_model': '../word_embedding_models/german/sent2vec/de_model.bin',
    'document_similarity': True,
    'document_similarity_new_candidate_constant': 0.1,
    'document_similarity_weights': (1.0, 0.1),
    # 'global_covariance': False,
    # 'global_covariance_weights': (4.0, 0.1),

    # 'filter_reference_keyphrases': True # ONLY USE FOR KEYCLUSTER CHECKING!,
    # 'draw_graphs': True,
    'print_document_scores': False,

    'split': 'dev',
    # 'num_documents': 200,
    # 'batch_size': 100,
    'reference_table': 'stemmed_filtered_stemmed',
    # 'table': 'pos_tags',
    'write_to_db': False
}


def heise_eval():
    extractor = KeyphraseExtractor()
    models = [
        # KeyCluster,
        EmbedRank,
        # TfIdf,
        # TextRank,
        # SingleRank,
        # TopicRank,
        # MultipartiteRank
    ]

    # print("Computing the document frequency file.")
    # compute_db_document_frequency("heise_train_df_counts.tsv.gz", **kwargs)

    # print("Computing the global cooccurrence matrix.")
    # compute_global_cooccurrence("heise_train.cooccurrence", **kwargs)

    for m in models:
        print("Computing the F-Score for the Heise Dataset with {}".format(m))
        evaluators = extractor.calculate_model_f_score(m, input_data='heise', **kwargs)
        print("\n\n")
        for key, evaluator_data in evaluators.items():
            macro_precision = evaluator_data['macro_precision']
            macro_recall = evaluator_data['macro_recall']
            macro_f_score = evaluator_data['macro_f_score']

            print("%s %s %s" % (str(macro_precision).replace('.', ','), str(macro_recall).replace('.', ','),
                                str(macro_f_score).replace('.', ',')))
            # print("%s - Macro average precision: %s, recall: %s, f-score: %s" % (
            # key, macro_precision, macro_recall, macro_f_score))


def collect_dataset_statistics():
    # some dataset statistic collection
    from common.DatabaseHandler import DatabaseHandler

    keyphrase_extractor = KeyphraseExtractor()
    db_handler = DatabaseHandler()

    documents, reference_stemmed = db_handler.load_split_from_db(KeyCluster, dataset='heise', **kwargs)

    unique_keyphrases = set()
    num_keyphrases_found = 0
    num_keyphrases_not_found = 0
    docs = 0
    keyphrase_length = 0
    for filename, doc in documents.items():
        doc = doc['document']
        docs += 1
        reference = reference_stemmed[filename]
        unique_keyphrases.update(reference)
        for keyphrase in reference:
            keyphrase_length += len(keyphrase.split(' '))
            tmp = False
            for sent in doc.sentences:
                if keyphrase_extractor._is_exact_match(keyphrase.lower(), ' '.join(sent.words).lower()):
                    num_keyphrases_found +=1
                    tmp = True
                    break
            if tmp is False:
                num_keyphrases_not_found += 1
                for sent in doc.sentences:
                    print(sent.words)
                print(keyphrase, filename)
    print("#Keyphrases found in text: %s" % num_keyphrases_found)
    print("#Keyphrases not found in text: %s" % num_keyphrases_not_found)
    print("#Unique Keyphrases: %s" % len(unique_keyphrases))
    print("#Documents: %s" % docs)
    x = 0
    for reference in reference_stemmed.values():
        x += len(reference)
    print("#Keyphrases total: %s" % x)
    keyphrase_length = keyphrase_length / x
    print("Average keyphrase length: %s" % keyphrase_length)
    print("Keyphrases found in text: %s" % ((100/x)*num_keyphrases_found))
    print("Avg #Keyphrases: %s" % (x/docs))



def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    heise_eval()
    # collect_dataset_statistics()

if __name__ == '__main__':
    main()

