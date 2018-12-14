from KeyCluster import KeyCluster
from main import KeyphraseExtractor
import pke
from helper import custom_normalize_POS_tags
from main import kwargs
from ClusterFeatureCalculator import CooccurrenceClusterFeature, WordEmbeddingsClusterFeature
from Cluster import HierarchicalClustering, SpectralClustering
import numpy as np
from evaluation import stemmed_wordwise_phrase_compare, stemmed_compare
import csv


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
    'DUC-2001': {
        'train': "../ake-datasets/datasets/DUC-2001/train",
        'test': "../ake-datasets/datasets/DUC-2001/test",
        'reference_stemmed': "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json",
        'reference_unstemmed': "../ake-datasets/datasets/DUC-2001/references/test.reader.json"
    }
}


all_parameter_options = {
    # 'language': ['en'],
    'normalization': ["stemming", None],
    # 'n_grams': [1],
    'window': list(np.linspace(2,10, 5, dtype=int)),
    'method': ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'],
    # 'candidate_selector': CandidateSelector(key_cluster_candidate_selector),
    'cluster_feature_calculator': [CooccurrenceClusterFeature, WordEmbeddingsClusterFeature],
    'cluster_method': [HierarchicalClustering, SpectralClustering],
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
    'evaluator_compare_func': [stemmed_compare, stemmed_wordwise_phrase_compare]
}


def evaluate_parameters(all_parameter_options, standard_parameters):
    extractor = KeyphraseExtractor()
    model = KeyCluster
    csv_file = ""

    with open('parameter_evaluation.csv', 'w') as parameter_evaluation_file:
        csv_writer = csv.writer(parameter_evaluation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Parameter', 'Value', 'Comp Func', 'Precision', 'Recall', 'F-Score'])

        for dataset, meta in _DATASETS.items():
            print("\tEvaluating various parameters for the %s Dataset." % (dataset))

            parameter_count = 0
            for eval_comp_func in all_parameter_options.get('evaluator_compare_func', [stemmed_compare]):
                print("Evaluating with evaluator compare func: %s" % eval_comp_func)
                standard_parameters.update({'evaluator_compare_func': eval_comp_func})

                for parameter, values in all_parameter_options.items():
                    updated_parameters = standard_parameters.copy()
                    if parameter == 'evaluator_compare_func':
                        continue

                    for val in values:
                        updated_parameters.update({parameter: val})
                        print("\nEvaluating with the following parameter: %s; value: %s" % (parameter, val))
                        parameter_count += 1

                        if standard_parameters['normalization'] == 'stemming':
                            reference = pke.utils.load_references(meta['reference_stemmed'])
                        else:
                            reference = pke.utils.load_references(meta['reference_unstemmed'])

                        macro_precision, macro_recall, macro_f_score = extractor.calculate_model_f_score(model, meta['test'],
                                                                                                         reference,
                                                                                                         print_document_scores=False,
                                                                                                         **updated_parameters)
                        print("\tMacro average precision: %s, recall: %s, f-score: %s" % (
                            macro_precision, macro_recall, macro_f_score))
                        csv_writer.writerow([parameter, val, eval_comp_func, macro_precision, macro_recall, macro_f_score])

            print("Finished evaluating %s different parameter options" % (parameter_count))


def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    evaluate_parameters(all_parameter_options, kwargs)


if __name__ == '__main__':
    main()
