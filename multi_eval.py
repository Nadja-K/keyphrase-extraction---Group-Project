from KeyCluster import KeyCluster
from KeyphraseExtractor import KeyphraseExtractor
import pke
from helper import custom_normalize_POS_tags
from KeyphraseExtractor import kwargs
from ClusterFeatureCalculator import CooccurrenceClusterFeature, WordEmbeddingsClusterFeature
from Cluster import HierarchicalClustering, SpectralClustering
import numpy as np
from evaluation import stemmed_wordwise_phrase_compare, stemmed_compare
import csv
import itertools


_DATASETS = {
    'Inspec': {
        'train': "../ake-datasets/datasets/Inspec/train",
        'test': "../ake-datasets/datasets/Inspec/dev",
        'reference_stemmed': "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json",
        'reference_unstemmed': "../ake-datasets/datasets/Inspec/references/dev.uncontr.json"
    },
    'SemEval-2010': {
        'train':  "../ake-datasets/datasets/SemEval-2010/train",
        'test': "../ake-datasets/datasets/SemEval-2010/test",
        'reference_stemmed': "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json",
        'reference_unstemmed': "../ake-datasets/datasets/SemEval-2010/references/test.combined.json",
    },
    # 'DUC-2001': {
    #     'train': "../ake-datasets/datasets/DUC-2001/train",
    #     'test': "../ake-datasets/datasets/DUC-2001/test",
    #     'reference_stemmed': "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json",
    #     'reference_unstemmed': "../ake-datasets/datasets/DUC-2001/references/test.reader.json"
    # }
}


all_parameter_options = {
    'normalization': ["stemming", None],
    # 'n_grams': [1],
    'window': list(np.linspace(2, 10, 5, dtype=int)),
    'method': ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'],
    # 'candidate_selector': CandidateSelector(key_cluster_candidate_selector),
    'cluster_method': [HierarchicalClustering],#, SpectralClustering],
    # 'keyphrase_selector': ,
    # 'regex': ['a*n+'],
    # 'num_clusters': ,
    # 'cluster_calc': ,
    'factor': [1/4, 1/3, 1/2, 2/3, 4/5],
    'frequent_word_list_file': [None, 'data/frequent_word_lists/en_50k.txt'],
    'min_word_count': [1000, 5000, 10000],
    # 'frequent_word_list': [['test'], []],
    'word_embedding_model_file': ['../word_embedding_models/english/Wikipedia2014_Gigaword5/la_vectors_glove_6b_50d'],
    # 'word_embedding_model':
}
parameter_combination_options = {
    'cluster_feature_calculator': [CooccurrenceClusterFeature, WordEmbeddingsClusterFeature],
}


def evaluate_parameter_combinations(all_parameter_options, parameter_combination_options, standard_parameters):
    for dataset, meta in _DATASETS.items():
        header_row = []
        with open(dataset + '_parameter_evaluation.csv', 'w') as parameter_evaluation_file:
            csv_writer = csv.writer(parameter_evaluation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            print("\tEvaluating various parameters for the %s Dataset." % (dataset))
            runs_total_per_dataset = 0
            write_header_row = True

            keys, values = zip(*parameter_combination_options.items())
            for v in itertools.product(*values):
                first_values = []
                experiment = dict(zip(keys, v))

                # Create the header row
                for i, param in enumerate(experiment):
                    if len(header_row) == 0:
                        header_row.append(param)

                    if experiment[param] == CooccurrenceClusterFeature:
                        first_values.append("Cooccurrence")
                    elif experiment[param] == WordEmbeddingsClusterFeature:
                        first_values.append("Word Embeddings")
                    else:
                        first_values.append(experiment[param])

                standard_parameters.update(experiment)
                print("\n\nEvaluating with the following combination parameters: %s" % experiment)
                runs_total_per_dataset += evaluate_each_parameter(all_parameter_options, standard_parameters, meta, csv_writer, header_row=header_row, first_values=first_values, write_header_row=write_header_row)
                write_header_row = False

            print("Finished evaluating %s different parameter options" % (runs_total_per_dataset))


def evaluate_each_parameter(all_parameter_options, standard_parameters, meta, csv_writer, header_row=[], first_values=[], write_header_row=True):
    extractor = KeyphraseExtractor()
    model = KeyCluster
    if write_header_row is True:
        header_row += ['Parameter', 'Value']
        for eval_comp_func in standard_parameters['evaluator_compare_func']:
            header_row += [eval_comp_func.__name__ + ' Precision', eval_comp_func.__name__ + ' Recall', eval_comp_func.__name__ + ' F-Score']
        csv_writer.writerow(header_row)

    num_runs_per_dataset = 0
    for parameter, values in all_parameter_options.items():

        updated_parameters = standard_parameters.copy()
        if parameter == 'evaluator_compare_func':
            continue

        for val in values:
            updated_parameters.update({parameter: val})
            print("\nEvaluating with the following parameter: %s; value: %s" % (parameter, val))
            num_runs_per_dataset += 1
            csv_row = first_values + [parameter, val]

            if standard_parameters['normalization'] == 'stemming':
                reference = pke.utils.load_references(meta['reference_stemmed'])
            else:
                reference = pke.utils.load_references(meta['reference_unstemmed'])

            evaluators = extractor.calculate_model_f_score(model, meta['test'], reference, print_document_scores=False, **updated_parameters)
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

    evaluate_parameter_combinations(all_parameter_options, parameter_combination_options, kwargs)


if __name__ == '__main__':
    main()
