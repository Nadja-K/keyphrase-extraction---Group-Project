from common.CandidateSelector import CandidateSelector, key_cluster_candidate_selector
from eval.evaluation import word_compare, stemmed_compare, wordwise_phrase_compare, stemmed_wordwise_phrase_compare
from methods.KeyCluster import KeyCluster
from common.KeyphraseExtractor import KeyphraseExtractor
import pke
from common.helper import custom_normalize_POS_tags
from common.ClusterFeatureCalculator import CooccurrenceClusterFeature, WordEmbeddingsClusterFeature
from common.Cluster import HierarchicalClustering
import numpy as np
import csv
import itertools
import sklearn


_DATASETS = {
    # 'Inspec': {
    #     'train': "../ake-datasets/datasets/Inspec/train",
    #     'test': "../ake-datasets/datasets/Inspec/test",
    #     'reference_stemmed': "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json",
    #     'reference_unstemmed': "../ake-datasets/datasets/Inspec/references/test.uncontr.json"
    # },
    # 'SemEval-2010': {
    #     'train':  "../ake-datasets/datasets/SemEval-2010/train",
    #     'test': "../ake-datasets/datasets/SemEval-2010/test",
    #     'reference_stemmed': "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json",
    #     'reference_unstemmed': "../ake-datasets/datasets/SemEval-2010/references/test.combined.json",
    # },
    # 'DUC-2001': {
    #     'train': "../ake-datasets/datasets/DUC-2001/train",
    #     'test': "../ake-datasets/datasets/DUC-2001/test",
    #     'reference_stemmed': "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json",
    #     'reference_unstemmed': "../ake-datasets/datasets/DUC-2001/references/test.reader.json"
    # }
    'Heise': {
        'split': 'dev'
    }
}

standard_parameter_options = {
    'language': 'en',
    'normalization': "stemming",
    'window': 2,
    'method': 'ward',
    'candidate_selector': CandidateSelector(key_cluster_candidate_selector),
    'cluster_method': HierarchicalClustering,
    'regex': 'a*n+',
    # 'cluster_calc': ,
    'factor': 2/3,
    'frequent_word_list_file': 'data/frequent_word_lists/en_50k.txt',
    'min_word_count': 1000,
    'word_embedding_model_file': '../word_embedding_models/english/Wikipedia2014_Gigaword5/la_vectors_glove_6b_50d',
    'cluster_feature_calculator': CooccurrenceClusterFeature, #WordEmbeddingsClusterFeature,#PPMIClusterFeature,
    'word_embedding_comp_func': sklearn.metrics.pairwise.cosine_similarity,#np.dot,
    'evaluator_compare_func': [stemmed_compare, stemmed_wordwise_phrase_compare],
    'print_document_scores': False,
    'write_to_db': False
}

changed_parameter_options = [
    {
        'window': 2,
    },
    # {
    #     'window': 4
    # },
    {
        'cluster_feature_calculator': WordEmbeddingsClusterFeature
    }
]


def evaluate_parameter_combinations(standard_parameter_options, parameter_combination_options):
    for dataset, meta in _DATASETS.items():
        header_row = []
        with open(dataset + '_parameter_evaluation.csv', 'w') as parameter_evaluation_file:
            csv_writer = csv.writer(parameter_evaluation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            print("\tEvaluating various parameters for the %s Dataset." % (dataset))
            runs_total_per_dataset = 0

            # Write the header row
            for key in standard_parameter_options:
                header_row.append(key)

            for eval_comp_func in standard_parameter_options['evaluator_compare_func']:
                header_row += [eval_comp_func.__name__ + ' Precision', eval_comp_func.__name__ + ' Recall',
                               eval_comp_func.__name__ + ' F-Score']

            csv_writer.writerow(header_row)

            for changed_parameters in parameter_combination_options:
                updated_parameters = standard_parameter_options.copy()
                updated_parameters.update(changed_parameters)

                print("\n\nEvaluating with the following combination parameters: %s" % changed_parameters)
                runs_total_per_dataset += evaluate_each_parameter(updated_parameters, meta, csv_writer)

            print("Finished evaluating %s different parameter options" % (runs_total_per_dataset))


def evaluate_each_parameter(updated_parameters, meta, csv_writer):
    extractor = KeyphraseExtractor()
    model = KeyCluster

    num_runs_per_dataset = 0
    csv_row = []
    for key, setting in updated_parameters.items():
        csv_row.append(setting)

    # print(updated_parameters)
    print(csv_row)
    if len(meta.keys()) == 1:
        updated_parameters['split'] = meta['split']
        evaluators = extractor.calculate_model_f_score(model, 'heise', None, **updated_parameters)
    else:
        if updated_parameters['normalization'] == 'stemming':
            reference = pke.utils.load_references(meta['reference_stemmed'])
        else:
            reference = pke.utils.load_references(meta['reference_unstemmed'])

        evaluators = extractor.calculate_model_f_score(model, meta['test'], reference, **updated_parameters)

    for key, evaluator_data in evaluators.items():
        macro_precision = evaluator_data['macro_precision']
        macro_recall = evaluator_data['macro_recall']
        macro_f_score = evaluator_data['macro_f_score']

        print("%s - Macro average precision: %s, recall: %s, f-score: %s" % (key, macro_precision, macro_recall, macro_f_score))
        csv_row += [macro_precision, macro_recall, macro_f_score]

    csv_writer.writerow(csv_row)

    return num_runs_per_dataset

def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    evaluate_parameter_combinations(standard_parameter_options, changed_parameter_options)


if __name__ == '__main__':
    main()
