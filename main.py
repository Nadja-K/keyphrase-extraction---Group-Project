import pke
import spacy
from nltk.corpus import stopwords
from pke import compute_document_frequency, compute_lda_model
from string import punctuation
import glob
import os
from pke.unsupervised import (
    TopicRank, SingleRank,
    MultipartiteRank, PositionRank,
    TopicalPageRank, ExpandRank,
    TextRank, TfIdf, KPMiner,
    YAKE, FirstPhrases
)
from KeyCluster import KeyCluster
from ClusterFeatureCalculator import CooccurrenceClusterFeature, PPMIClusterFeature, WordEmbeddingsClusterFeature
from CandidateSelector import CandidateSelector
from Cluster import HierarchicalClustering, SpectralClustering
from KeyphraseSelector import KeyphraseSelector
from evaluation import Evaluator, stemmed_wordwise_phrase_compare, stemmed_compare, stemmed_word_compare
from Cluster import euclid_dist
from helper import compute_df, calc_num_cluster, custom_normalize_POS_tags, _load_word_embedding_model, _load_frequent_word_list


class KeyphraseExtractor:
    def extract_keyphrases(self, model, file, **params):
        language = params.get('language', 'en')
        normalization = params.get('normalization', 'stemming')
        frequency_file = params.get('frequency_file', None)
        redundancy_removal = params.get('redundancy_removal', False)

        extractor = model()
        extractor.load_document(file, language=language, normalization=normalization)

        df = None
        if frequency_file is not None:
            df = pke.load_document_frequency_file(input_file=frequency_file)

        if model in [TfIdf]:
            """
            :param list stoplist
            :param int n_grams
            :param str frequency_file
            """
            stoplist = params.get('stoplist', extractor.stoplist)
            n_grams = params.get('n_grams', 3)

            extractor.candidate_selection(n=n_grams, stoplist=stoplist)
            extractor.candidate_weighting(df=df)

        elif model in [TextRank]:
            """
            :param int window
            :param set pos
            :param float top_percent
            :param bool normalized
            :param bool run_candidate_selection
            """
            window = params.get('window', 2)
            pos = params.get('pos', ('NOUN', 'PROPN', 'ADJ'))
            top_percent = params.get('top_percent', 1.0)
            normalized = params.get('normalized', False)

            if params.get('run_candidate_selection', False):
                extractor.candidate_selection(pos=pos)
            extractor.candidate_weighting(window=window, pos=pos, top_percent=top_percent, normalized=normalized)

        elif model in [SingleRank]:
            """
            :param set pos
            :param int window
            """
            window = params.get('window', 10)
            pos = params.get('pos', ('NOUN', 'PROPN', 'ADJ'))

            extractor.candidate_selection(pos=pos)
            extractor.candidate_weighting(window=window, pos=pos)

        elif model in [TopicRank]:
            """
            :param list stoplist
            :param set pos
            :param float threshold
            :param str method
            :param str heuristic
            """
            pos = params.get('pos', ('NOUN', 'PROPN', 'ADJ'))
            stoplist = params.get('stoplist', list(punctuation) + ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
                                  + stopwords.words(pke.base.ISO_to_language[params.get('language', 'en')]))
            threshold = params.get('threshold', 0.74)
            method = params.get('method', 'average')
            heuristic = params.get('heuristic', None)
            extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_weighting(threshold=threshold, method=method, heuristic=heuristic)

        elif model in [MultipartiteRank]:
            """
            :param list stoplist
            :param set pos
            :param float threshold
            :param str method
            :param float alpha
            """
            pos = params.get('pos', ('NOUN', 'PROPN', 'ADJ'))
            stoplist = params.get('stoplist', list(punctuation) + ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
                                  + stopwords.words(pke.base.ISO_to_language[params.get('language', 'en')]))
            threshold = params.get('threshold', 0.74)
            method = params.get('method', 'average')
            alpha = params.get('alpha', 1.1)

            extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_weighting(threshold=threshold, method=method, alpha=alpha)

        elif model in [TopicalPageRank]:
            """
            :param str grammar
            :param int window
            :param set pos
            :param pickle.gz lda_model
            :param list stoplist
            :param bool normalized
            """
            grammar = params.get('grammar', "NP:{<ADJ>*<NOUN|PROPN>+}")
            window = params.get('window', 10)
            pos = params.get('pos', ('NOUN', 'PROPN', 'ADJ'))
            stoplist = params.get('stoplist', None)
            normalized = params.get('normalized', False)
            lda_model = params.get('lda_model', None)

            extractor.candidate_selection(grammar=grammar)
            extractor.candidate_weighting(window=window, pos=pos, lda_model=lda_model, stoplist=stoplist, normalized=normalized)

        elif model in [PositionRank]:
            """
            :param str grammar
            :param int maximum_word_number
            :param int window
            :param set pos
            :param bool normalized
            """
            grammar = params.get('grammar', "NP:{<ADJ>*<NOUN|PROPN>+}")
            max_word_num = params.get('maximum_word_number', 3)
            window = params.get('window', 10)
            pos = params.get('pos', ('NOUN', 'PROPN', 'ADJ'))
            normalized = params.get('normalized', False)

            extractor.candidate_selection(grammar=grammar, maximum_word_number=max_word_num)
            extractor.candidate_weighting(window=window, pos=pos, normalized=normalized)

        elif model in [YAKE]:
            """
            :param int n_grams
            :param list stoplist
            :param int window
            """
            n_grams = params.get('n_grams', 3)
            stoplist = params.get('stoplist', None)
            window = params.get('window', 2)

            extractor.candidate_selection(n=n_grams, stoplist=stoplist)
            extractor.candidate_weighting(window=window, stoplist=stoplist, use_stems=(normalization == 'stemming'))

        elif model in [KPMiner]:
            """
            :param int lasf
            :param int cutoff
            :param list stoplist
            :param float sigma
            :param float alpha 
            :param str frequency_file
            """
            lasf = params.get('lasf', 3)
            cutoff = params.get('cutoff', 400)
            stoplist = params.get('stoplist', None)
            sigma = params.get('sigma', 3.0)
            alpha = params.get('alpha', 2.3)

            extractor.candidate_selection(lasf=lasf, cutoff=cutoff, stoplist=stoplist)
            extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma, encoding="utf-8")

        elif model in [KeyCluster]:
            """
            :param CandidateSelector candidate_selector
            :param ClusterFeatureCalculator cluster_feature_calculator
            :param Cluster cluster_method
            :param func exemplar_dist_func
            :param KeyphraseSelector keyphrase_selector
            :param int window
            :param int num_clusters
            :param func cluster_calc
            :param float factor
            :param str regex
            :param str method
            :param bool transformToDistanceMatrix
            :param spacy instanz word_embedding_model 
            :param list frequent_word_list
            """
            candidate_selector = params.get('candidate_selector', CandidateSelector())

            cluster_feature_calculator = params.get('cluster_feature_calculator', CooccurrenceClusterFeature)
            cluster_feature_calculator = cluster_feature_calculator(**params)
            cluster_method = params.get('cluster_method', HierarchicalClustering)
            cluster_method = cluster_method(**params)

            exemplar_terms_dist_func = params.get('exemplar_terms_dist_func', euclid_dist)

            keyphrase_selector = params.get('keyphrase_selector', KeyphraseSelector())
            regex = params.get('regex', 'a*n+')

            frequent_word_list = params.get('frequent_word_list', [])
            if len(frequent_word_list) == 0:
                print("Frequent word list is empty. No frequent word filtering will be performed.")

            # Cluster Candidate Selection
            extractor.candidate_selection(candidate_selector=candidate_selector, **params)

            # Calculate number of Clusters (if the cluster algorithm needs this)
            num_clusters = params.get('num_clusters', 0)
            cluster_calc = params.get('cluster_calc', calc_num_cluster)
            factor = params.get('factor', 2/3)
            cluster_calc_args = {
                'factor': factor,
                'context': extractor,
                'num_clusters': num_clusters
            }
            num_clusters = cluster_calc(**cluster_calc_args)

            # Candidate Clustering, Exemplar Term Selection, Keyphrase Selection
            num_clusters = extractor.candidate_weighting(cluster_feature_calculator=cluster_feature_calculator,
                                          cluster_method=cluster_method,
                                          exemplar_terms_dist_func=exemplar_terms_dist_func,
                                          keyphrase_selector=keyphrase_selector,
                                          num_clusters=num_clusters,
                                          filename=file,
                                          regex=regex,
                                          frequent_word_list=frequent_word_list)
        else:
            extractor.candidate_selection()
            extractor.candidate_weighting()

        n_keyphrases = params.get('n_keyphrases', len(extractor.candidates))
        return extractor.get_n_best(n=n_keyphrases, redundancy_removal=redundancy_removal, stemming=(normalization == 'stemming')), extractor

    def calculate_model_f_score(self, model, input_dir, references, print_document_scores=True, **kwargs):
        precision_total = 0
        recall_total = 0
        f_score_total = 0
        num_documents = 0

        # Parse the frequent word list once for the model
        kwargs = _load_frequent_word_list(**kwargs)

        # Load a spacy model once for the model
        kwargs = _load_word_embedding_model(**kwargs)

        eval_comp_func = kwargs.get('evaluator_compare_func', stemmed_compare)
        for file in glob.glob(input_dir + '/*'):
            # make sure to initialize the evaluator clean for every run!
            evaluator = Evaluator()
            evaluator.compare_func = eval_comp_func

            filename = os.path.splitext(os.path.basename(file))[0]
            reference = references[filename]
            if print_document_scores is True:
                print("Processing File: %s" % filename)

            keyphrases, context = self.extract_keyphrases(model, file, **kwargs)
            # Filter out reference keyphrases that don't appear in the original text
            if kwargs.get('filter_reference_keyphrases', False) is True:
                reference = self._filter_reference_keyphrases(reference, context, kwargs.get('normalization', 'stemming'))

            if (len(keyphrases) > 0 and len(reference) > 0):
                evaluator.evaluate(reference, list(zip(*keyphrases))[0])
                # print(list(zip(*keyphrases))[0])
                # print(reference)
                if print_document_scores is True:
                    print("Precision: %s, Recall: %s, F-Score: %s" % (evaluator.precision, evaluator.recall, evaluator.f_measure))
                precision_total += evaluator.precision
                recall_total += evaluator.recall
                f_score_total += evaluator.f_measure
            else:
                print("Skipping file %s for not enough reference keyphrases or found keyphrases. Found keyphrases: %s, Reference keyphrases: %s" % (filename, len(keyphrases), len(reference)))
            num_documents += 1

        macro_precision = (precision_total / num_documents)
        macro_recall = (recall_total / num_documents)
        macro_f_score = (f_score_total / num_documents)
        return macro_precision, macro_recall, macro_f_score

    def window_generator(self, iterable, size):
        for i in range(len(iterable) - size + 1):
            yield iterable[i:i + size]

    def _is_exact_match(self, substring, string):
        substring_tokens = substring.split(' ')
        string_tokens = string.split(' ')

        for string_window in self.window_generator(string_tokens, len(substring_tokens)):
            if substring_tokens == string_window:
                return True

        return False

    def _filter_reference_keyphrases(self, reference_keyphrases, model, normalization):
        filtered_reference_keyphrases = []

        for keyphrase in reference_keyphrases:
            keyphrase = keyphrase.translate(str.maketrans({"(": "-lrb- ",
                                                           ")": " -rrb-",
                                                           "{": r"-lcb- ",
                                                           "}": r" -rcb-"
                                                           }))
            for s in model.sentences:
                if normalization == 'stemming':
                    sentence = ' '.join(s.stems)
                else:
                    sentence = ' '.join(s.words)

                if self._is_exact_match(keyphrase, sentence):
                    filtered_reference_keyphrases.append(keyphrase)
                    break
        filtered_reference_keyphrases = set(filtered_reference_keyphrases)
        return filtered_reference_keyphrases


kwargs = {
    'language': 'en',
    'normalization': "stemming",
    # 'n_keyphrases': 30,
    # 'redundancy_removal': ,
    # 'n_grams': 1,
    # 'stoplist': ,
    # 'frequency_file': '../ake-datasets/datasets/Inspec/Inspec_df_counts.tsv.gz',
    'window': 2,
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
    # 'cluster_feature_calculator': WordEmbeddingsClusterFeature,#WordEmbeddingsClusterFeature,
    # 'cluster_method': SpectralClustering,
    # 'keyphrase_selector': ,
    # 'regex': 'a*n+',
    # 'num_clusters': ,
    # 'cluster_calc': ,
    # 'factor': 2/3,
    'frequent_word_list_file': 'data/frequent_word_lists/en_50k.txt',
    'min_word_count': 1000,
    # 'frequent_word_list': ['test'],
    'word_embedding_model_file': '../word_embedding_models/english/Wikipedia2014_Gigaword5/la_vectors_glove_6b_50d',
    # 'word_embedding_model':
    # 'evaluator_compare_func': stemmed_wordwise_phrase_compare, #stemmed_wordwise_phrase_compare,
    # 'filter_reference_keyphrases': True # ONLY USE FOR KEYCLUSTER CHECKING!
}

def custom_testing():
    # SemEval-2010
    # train_folder = "../ake-datasets/datasets/SemEval-2010/train"
    # test_folder = "../ake-datasets/datasets/SemEval-2010/test"
    # reference_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json"

    # Inspec
    train_folder = "../ake-datasets/datasets/Inspec/train"
    test_folder = "../ake-datasets/datasets/Inspec/dev"
    reference_stemmed_file = "../ake-datasets/datasets/Inspec/references/dev.uncontr.stem.json"

    # Heise
    ## train_folder = "../ake-datasets/datasets/Inspec/train"
    # test_folder = "../parsed"
    # reference_stemmed_file = ""

    # DUC-2001
    # train_folder = "../ake-datasets/datasets/DUC-2001/train"
    # test_folder = "../ake-datasets/datasets/DUC-2001/test"
    # reference_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json"

    reference_stemmed = pke.utils.load_references(reference_stemmed_file)
    extractor = KeyphraseExtractor()
    models = [
        KeyCluster,
        # TfIdf,
        # TopicRank,
        # SingleRank,
        # TextRank,
        # KPMiner
    ]

    for m in models:
        if m == TfIdf:
            frequency_file = kwargs.get('frequency_file', None)
            if frequency_file is None:
                output_name = '/'.join(train_folder.split('/')[:-1]) + '/df_counts.tsv.gz'
                compute_df(train_folder, output_name, extension="xml")
                kwargs['frequency_file'] = output_name

        print("Computing the F-Score for the Inspec Dataset with {}".format(m))
        macro_precision, macro_recall, macro_f_score = extractor.calculate_model_f_score(m, test_folder, reference_stemmed, **kwargs)
        print("Macro average precision: %s, recall: %s, f-score: %s" % (macro_precision, macro_recall, macro_f_score))

    # For testing with a single raw text file.
    # for m in models:
    #     keyphrases = extractor.extract_keyphrases(m, 'test_input.txt', **kwargs)
    #     print(keyphrases)

    # for m in models:
    #     i = 0
    #     for file in glob.glob(heise_folder + '/*'):
    #         if i >= 0:
    #             keyphrases = extractor.extract_keyphrases(m, file, **kwargs)
    #             print(keyphrases)
    #         i += 1
    #         if i == 2:
    #             break


def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    custom_testing()


if __name__ == '__main__':
    main()
