import os
import glob
import pke
from pke.unsupervised import (
    TfIdf, SingleRank, TextRank, TopicRank, MultipartiteRank
)
from methods.KeyCluster import KeyCluster
from methods.EmbedRank import EmbedRank

from common.KeyphraseExtractor import KeyphraseExtractor

from eval.evaluation import stemmed_wordwise_phrase_compare, stemmed_compare

from common.helper import custom_normalize_POS_tags, compute_df

pke.base.ISO_to_language['de'] = 'german'
pke.LoadFile.normalize_pos_tags = custom_normalize_POS_tags


kwargs = {
    # 'language': 'en',
    'normalization': "stemming",
    'n_keyphrases': 10,
    # 'redundancy_removal': ,
    # 'n_grams': 1,
    # 'stoplist': ,
    # 'frequency_file': '../ake-datasets/datasets/SemEval-2010/df_counts.tsv.gz',
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
    # 'cluster_feature_calculator': WordEmbeddingsClusterFeature,#PPMIClusterFeature,
    # word_embedding_comp_func': sklearn.metrics.pairwise.cosine_similarity,#np.dot,
    # 'global_cooccurrence_matrix': 'inspec_out.cooccurrence',#'semeval_out.cooccurrence',
    # 'cluster_method': SpectralClustering,
    # 'keyphrase_selector': ,
    # 'regex': '',
    # 'num_clusters': 20,
    # 'cluster_calc': ,
    # 'factor': 2/3,
    # 'frequent_word_list_file': 'data/frequent_word_lists/en_50k.txt',
    # 'min_word_count': 1000,
    # 'frequent_word_list': ['test'],
    # 'word_embedding_model_file': '../word_embedding_models/english/Wikipedia2014_Gigaword5/la_vectors_glove_6b_50d',
    # 'word_embedding_model':
    'evaluator_compare_func': [stemmed_compare, stemmed_wordwise_phrase_compare], #stemmed_wordwise_phrase_compare,

    ## EmbedRank
    'sent2vec_model': '../word_embedding_models/english/sent2vec/wiki_bigrams.bin',
    'document_similarity': False,
    'document_similarity_new_candidate_constant': 1.0,
    'global_covariance': False,
    'global_covariance_weights': (4.0, 0.1),

    # 'filter_reference_keyphrases': True # ONLY USE FOR KEYCLUSTER CHECKING!,
    # 'draw_graphs': True,
    # 'print_document_scores': False,

    # 'num_documents': 200,
    # 'batch_size': 100,
    # 'reference_table': 'stemmed_filtered_stemmed',
    # 'table': 'pos_tags',
    'write_to_db': False
}


def custom_testing():
    # SemEval-2010
    train_folder = "../ake-datasets/datasets/SemEval-2010/train"
    test_folder = "../ake-datasets/datasets/SemEval-2010/test"
    # reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json"

    # Only needs to be done once for a dataset
    # print("Computing the global cooccurrence matrix.")
    # compute_document_cooccurrence(test_folder, "semeval_out.cooccurrence", **kwargs)
    # compute_global_cooccurrence(test_folder, "semeval_out.cooccurrence", **kwargs)

    # Inspec
    train_folder = "../ake-datasets/datasets/Inspec/train"
    # test_folder = "../ake-datasets/datasets/Inspec/dev"
    test_folder = "../ake-datasets/datasets/Inspec/test"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json"
    reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json"

    # Only needs to be done once for a dataset
    # print("Computing the global cooccurrence matrix.")
    # compute_global_cooccurrence("inspec_out.cooccurrence", input_dir=test_folder, **kwargs)

    # DUC-2001
    # train_folder = "../ake-datasets/datasets/DUC-2001/train"
    # test_folder = "../ake-datasets/datasets/DUC-2001/test"
    # reference_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json"

    reference_stemmed = pke.utils.load_references(reference_stemmed_file)
    extractor = KeyphraseExtractor()
    models = [
        # KeyCluster,
        EmbedRank,
        # TfIdf,
        # TopicRank,
        # SingleRank,
        # TextRank,
        # MultipartiteRank,
        # KPMiner
    ]

    for m in models:
        if m == TfIdf:
            frequency_file = kwargs.get('frequency_file', None)
            if frequency_file is None:
                output_name = '/'.join(train_folder.split('/')[:-1]) + '/df_counts.tsv.gz'
                compute_df(train_folder, output_name, extension="xml")
                kwargs['frequency_file'] = output_name
                print("Frequency file calculated for current dataset.")

        print("Computing the F-Score for the Inspec Dataset with {}".format(m))
        evaluators = extractor.calculate_model_f_score(m, input_data=test_folder, references=reference_stemmed, **kwargs)
        print("\n\n")
        for key, evaluator_data in evaluators.items():
            macro_precision = evaluator_data['macro_precision']
            macro_recall = evaluator_data['macro_recall']
            macro_f_score = evaluator_data['macro_f_score']

            print("%s - Macro average precision: %s, recall: %s, f-score: %s" % (key, macro_precision, macro_recall, macro_f_score))

    # For testing with a single raw text file.
    # for m in models:
    #     keyphrases = extractor.extract_keyphrases(m, 'test_input.txt', **kwargs)
    #     print(keyphrases)

def extract_keyphrases_from_raw_text():
    extractor = KeyphraseExtractor()
    models = [
        # KeyCluster,
        # EmbedRank,
        # TfIdf,
        TopicRank,
        # SingleRank,
        # TextRank,
        # KPMiner
    ]

    for m in models:
        keyphrases = extractor.extract_keyphrases_from_raw_text(m, 'test_input.txt', **kwargs)
        print(keyphrases)

def collect_dataset_statistics():
    # some dataset statistic collection, can be removed
    train_folder = "../ake-datasets/datasets/DUC-2001/train"
    test_folder = "../ake-datasets/datasets/DUC-2001/test"
    reference_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json"
    #
    # train_folder = "../ake-datasets/datasets/Inspec/train"
    # test_folder = "../ake-datasets/datasets/Inspec/dev"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json"

    # train_folder = "../ake-datasets/datasets/SemEval-2010/train"
    # test_folder = "../ake-datasets/datasets/SemEval-2010/test"
    # reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json"

    keyphrase_extractor = KeyphraseExtractor()
    reference_stemmed = pke.utils.load_references(reference_stemmed_file)

    unique_keyphrases = set()
    num_keyphrases_found = 0
    num_keyphrases_not_found = 0
    docs = 0
    keyphrase_length = 0
    for file in glob.glob(test_folder + '/*'):
        docs += 1
        filename = os.path.splitext(os.path.basename(file))[0]
        extractor = KeyCluster()
        extractor.load_document(file, language='en', normalization=False)
        reference = reference_stemmed[filename]
        unique_keyphrases.update(reference)
        # print(reference)
        for keyphrase in reference:
            keyphrase_length += len(keyphrase.split(' '))
            tmp = False
            for sent in extractor.sentences:
                # if keyphrase_extractor._is_exact_match(keyphrase, ' '.join(sent.stems)):
                if keyphrase in ' '.join(sent.stems):
                    num_keyphrases_found +=1
                    tmp = True
                    break
            if tmp is False:
                num_keyphrases_not_found += 1
                print(keyphrase, file)
    print(num_keyphrases_found)
    print(num_keyphrases_not_found)
    print(len(unique_keyphrases))
    print(docs)
    x = 0
    for reference in reference_stemmed.values():
        x += len(reference)
    print(x)
    keyphrase_length = keyphrase_length / x
    print(keyphrase_length)


def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    custom_testing()
    # extract_keyphrases_from_raw_text()

if __name__ == '__main__':
    main()
