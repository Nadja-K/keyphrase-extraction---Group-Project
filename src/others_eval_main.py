import os
import glob
import pke
from pke.unsupervised import (
    TfIdf, SingleRank, TextRank, TopicRank, MultipartiteRank
)

from common.CandidateSelector import CandidateSelector, key_cluster_candidate_selector
from methods.KeyCluster import KeyCluster
from methods.EmbedRank import EmbedRank

from common.KeyphraseExtractor import KeyphraseExtractor

from eval.evaluation import stemmed_wordwise_phrase_compare, stemmed_compare, word_compare, wordwise_phrase_compare

from common.helper import custom_normalize_POS_tags, compute_df, compute_global_cooccurrence

pke.base.ISO_to_language['de'] = 'german'
pke.LoadFile.normalize_pos_tags = custom_normalize_POS_tags


kwargs = {
    'language': 'en',
    'normalization': "stemming",
    'n_keyphrases': 10,
    # 'redundancy_removal': ,
    # 'n_grams': 1,
    # 'stoplist': ,
    # 'frequency_file': 'data/document_frequency/semEval_train_df_counts.tsv.gz',#'data/document_frequency/semEval_train_df_counts.tsv.gz',#'../ake-datasets/datasets/SemEval-2010/df_counts.tsv.gz',
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
    # 'global_cooccurrence_matrix': 'data/global_cooccurrence/inspec_train.cooccurrence',#'semeval_out.cooccurrence',
    # 'cluster_method': SpectralClustering,
    # 'keyphrase_selector': ,
    # 'regex': '',
    # 'num_clusters': 5,#20,
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
    # 'document_similarity': True,
    # 'document_similarity_new_candidate_constant': 1.0,
    # 'document_similarity_weights': (1.0, 0.5),
    # 'global_covariance': False,
    # 'global_covariance_weights': (4.0, 0.1),

    # 'filter_reference_keyphrases': True # ONLY USE FOR KEYCLUSTER CHECKING!,
    # 'draw_graphs': True,
    'print_document_scores': False,

    # 'num_documents': 200,
    # 'batch_size': 100,
    # 'reference_table': 'stemmed_filtered_stemmed',
    # 'table': 'pos_tags',
    'write_to_db': False
}


def custom_testing():
    # SemEval-2010
    # train_folder = "../ake-datasets/datasets/SemEval-2010/train"
    # test_folder = "../ake-datasets/datasets/SemEval-2010/test"
    # reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json"

    # Only needs to be done once for a dataset
    # print("Computing the global cooccurrence matrix.")
    # compute_global_cooccurrence("semEval_train.cooccurrence", input_dir=train_folder, **kwargs)
    # compute_df(train_folder, "semEval_train_df_counts.tsv.gz", extension="xml")


    # Inspec
    train_folder = "../ake-datasets/datasets/Inspec/train"
    test_folder = "../ake-datasets/datasets/Inspec/dev"
    reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json"
    reference_unstemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json"

    test_folder = "../ake-datasets/datasets/Inspec/test"
    reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json"
    reference_unstemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.json"

    # Only needs to be done once for a dataset
    # print("Computing the global cooccurrence matrix.")
    # compute_global_cooccurrence("inspec_train.cooccurrence", input_dir=train_folder, **kwargs)
    # compute_df(train_folder, "inspec_train_df_counts.tsv.gz", extension="xml")


    # DUC-2001
    # train_folder = "../ake-datasets/datasets/DUC-2001/test"
    # test_folder = "../ake-datasets/datasets/DUC-2001/test"
    # reference_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json"

    # Only needs to be done once for a dataset
    # print("Computing the global cooccurrence matrix.")
    # compute_global_cooccurrence("duc_test.cooccurrence", input_dir=train_folder, **kwargs)
    # compute_df(train_folder, "duc_test_df_counts.tsv.gz", extension="xml")


    if kwargs.get('normalization', 'stemming') == 'stemming':
        reference = pke.utils.load_references(reference_stemmed_file)
    else:
        reference = pke.utils.load_references(reference_unstemmed_file)
    extractor = KeyphraseExtractor()
    models = [
        KeyCluster,
        # EmbedRank,
        # TfIdf,
        # TextRank,
        # SingleRank,
        # TopicRank,
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

        print("Computing the F-Score for the current Dataset with {}".format(m))
        evaluators = extractor.calculate_model_f_score(m, input_data=test_folder, references=reference, **kwargs)
        print("\n\n")
        for key, evaluator_data in evaluators.items():
            macro_precision = evaluator_data['macro_precision']
            macro_recall = evaluator_data['macro_recall']
            macro_f_score = evaluator_data['macro_f_score']

            print("%s %s %s" % (str(macro_precision).replace('.', ','), str(macro_recall).replace('.', ','), str(macro_f_score).replace('.', ',')))
            # print("%s - Macro average precision: %s, recall: %s, f-score: %s" % (key, macro_precision, macro_recall, macro_f_score))

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
    # some dataset statistic collection
    # folder = "../ake-datasets/datasets/DUC-2001/test"
    # reference_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json"
    # reference_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.json"

    # folder = "../ake-datasets/datasets/Inspec/train"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/train.uncontr.stem.json"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/train.uncontr.json"
    # folder = "../ake-datasets/datasets/Inspec/dev"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.json"
    # folder = "../ake-datasets/datasets/Inspec/test"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json"
    # reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.json"

    # folder = "../ake-datasets/datasets/SemEval-2010/train"
    # reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/train.combined.stem.json"
    # reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/train.combined.json"
    folder = "../ake-datasets/datasets/SemEval-2010/test"
    reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json"

    keyphrase_extractor = KeyphraseExtractor()
    reference_stemmed = pke.utils.load_references(reference_stemmed_file)

    unique_keyphrases = set()
    num_keyphrases_found = 0
    num_keyphrases_not_found = 0
    docs = 0
    keyphrase_length = 0
    for file in glob.glob(folder + '/*'):
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
                # keyphrase_list = [x.lower() for x in keyphrase.split(' ')]
                # sent_list = [x.lower() for x in sent.stems]
                #
                # for i in range(len(sent_list) - len(keyphrase_list) - 1):
                #     comp_res = stemmed_compare([' '.join(sent_list[i:i+len(keyphrase_list)]).lower()], [keyphrase])
                #     if(comp_res.tp >= 1):
                #         # print(keyphrase, sent_list[i:i+len(keyphrase_list)])
                #         num_keyphrases_found +=1
                #         tmp = True
                #         break
                #
                # if tmp is True:
                #     break

                # keyphrase_list = [x.lower() for x in keyphrase.split(' ')]
                # sent_list = [x.lower() for x in sent.stems]
                #
                # for i in range(len(sent_list) - len(keyphrase_list) - 1):
                #     found = True
                #     for j, keyword in enumerate(keyphrase_list):
                #         if keyword in sent_list[i+j]:
                #             continue
                #         else:
                #             found = False
                #             break
                #
                #     if found == True:
                #         num_keyphrases_found += 1
                #         tmp = True
                #         break
                #
                # if tmp is True:
                #     break

                if keyphrase_extractor._is_exact_match(keyphrase.lower(), ' '.join(sent.words).lower()):
                # if keyphrase in ' '.join(sent.stems):
                    num_keyphrases_found +=1
                    tmp = True
                    break
            if tmp is False:
                num_keyphrases_not_found += 1
                print(keyphrase, file)
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

    custom_testing()
    # collect_dataset_statistics()
    # extract_keyphrases_from_raw_text()

if __name__ == '__main__':
    main()
