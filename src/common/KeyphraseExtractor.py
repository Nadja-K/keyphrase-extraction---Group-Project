import pke
from nltk.corpus import stopwords
from string import punctuation
import glob
import os
from pke.unsupervised import (
    TopicRank, SingleRank,
    MultipartiteRank, PositionRank,
    TopicalPageRank, TextRank, TfIdf, KPMiner,
    YAKE
)

from common.EmbeddingDistributor import EmbeddingDistributor
from methods.KeyCluster import KeyCluster
from methods.EmbedRank import EmbedRank

from common.ClusterFeatureCalculator import CooccurrenceClusterFeature
from common.CandidateSelector import CandidateSelector, embed_rank_candidate_selector
from common.Cluster import HierarchicalClustering
from common.KeyphraseSelector import KeyphraseSelector
from common.Cluster import euclid_dist
from common.helper import calc_num_cluster, _load_word_embedding_model, \
    _load_frequent_word_list, load_global_cooccurrence_matrix, collect_keyphrase_data, load_document_similarity_data, load_global_covariance_matrix
from common.DatabaseHandler import DatabaseHandler

from eval.evaluation import Evaluator, stemmed_compare


class KeyphraseExtractor:
    def get_param(self,  key, default_value, **params):
        value = params.get(key, default_value)
        params.update({key: value})

        return value, params

    def extract_keyphrases(self, model, extractor, filename, **params):
        language, params = self.get_param('language', 'en', **params)
        normalization, params = self.get_param('normalization', 'stemming', **params)
        frequency_file, params = self.get_param('frequency_file', None, **params)
        redundancy_removal, params = self.get_param('redundancy_removal', False, **params)

        params['model'] = model.__name__

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
            :param bool draw_graphs
            """
            candidate_selector, params = self.get_param('candidate_selector', CandidateSelector(), **params)

            cluster_feature_calculator, params = self.get_param('cluster_feature_calculator', CooccurrenceClusterFeature, **params)
            cluster_feature_calculator = cluster_feature_calculator(**params)
            cluster_method, params = self.get_param('cluster_method', HierarchicalClustering, **params)
            cluster_method = cluster_method(**params)

            exemplar_terms_dist_func, params = self.get_param('exemplar_terms_dist_func', euclid_dist, **params)

            keyphrase_selector, params = self.get_param('keyphrase_selector', KeyphraseSelector(), **params)
            regex, params = self.get_param('regex', 'a*n+', **params)

            frequent_word_list, params = self.get_param('frequent_word_list', [], **params)
            if len(frequent_word_list) == 0:
                print("Frequent word list is empty. No frequent word filtering will be performed.")

            # Cluster Candidate Selection
            extractor.candidate_selection(**params)

            # Calculate number of Clusters (if the cluster algorithm needs this)
            num_clusters, params = self.get_param('num_clusters', 0, **params)
            cluster_calc, params = self.get_param('cluster_calc', calc_num_cluster, **params)
            factor, params = self.get_param('factor', 2/3, **params)
            cluster_calc_args = {
                'factor': factor,
                'context': extractor,
                'num_clusters': num_clusters
            }
            num_clusters = cluster_calc(**cluster_calc_args)
            draw_graphs, params = self.get_param('draw_graphs', False, **params)

            # Candidate Clustering, Exemplar Term Selection, Keyphrase Selection
            num_clusters = extractor.candidate_weighting(cluster_feature_calculator=cluster_feature_calculator,
                                          cluster_method=cluster_method,
                                          exemplar_terms_dist_func=exemplar_terms_dist_func,
                                          keyphrase_selector=keyphrase_selector,
                                          num_clusters=num_clusters,
                                          filename=filename,
                                          regex=regex,
                                          frequent_word_list=frequent_word_list,
                                          draw_graphs=draw_graphs)
            params['num_clusters'] = num_clusters
        elif model in [EmbedRank]:
            """
            :param str language
            :param str regex
            :param CandidateSelector candidate_selector
            :param EmbeddingDistributor sent2vec_model
            :param bool draw_graphs  
            :param DataFrame document_similarity_data     
            :param float document_similarity_new_candidate_constant
            """
            # Initialize standard parameters for EmbedRank
            regex, params = self.get_param('regex', 'a*n+', **params)
            candidate_selector, params = self.get_param('candidate_selector', CandidateSelector(embed_rank_candidate_selector), **params)
            sent2vec_model = params.get('sent2vec_model', None)
            draw_graphs, params = self.get_param('draw_graphs', False, **params)
            document_similarity_data = params.get('document_similarity_data', None)
            document_similarity_weights = params.get('document_similarity_weights', None)
            document_similarity_new_candidate_constant, params = self.get_param('document_similarity_new_candidate_constant', 1.0, **params)
            global_covariance_matrix = params.get('global_covariance_matrix', None)
            global_embedding_centroid = params.get('global_embedding_centroid', None)
            global_covariance_weights = params.get('global_covariance_weights', None)

            # Candidate Selection
            extractor.candidate_selection(**params)

            # Keyphrase Selection
            extractor.candidate_weighting(sent2vec_model=sent2vec_model, filename=filename, draw_graphs=draw_graphs,
                                          language=language, document_similarity_data=document_similarity_data,
                                          document_similarity_new_candidate_constant=document_similarity_new_candidate_constant,
                                          document_similarity_weights=document_similarity_weights,
                                          global_covariance_matrix=global_covariance_matrix,
                                          global_embedding_centroid=global_embedding_centroid,
                                          global_covariance_weights=global_covariance_weights)
        else:
            extractor.candidate_selection()
            extractor.candidate_weighting()

        n_keyphrases, params = self.get_param('n_keyphrases', len(extractor.candidates), **params)
        # Limit the number of keyphrases to be extracted based on the amount of candidates and a factor, KeyCluster
        # has this already implemented through the Clustering, so this model is excluded here.
        if model not in [KeyCluster]:
            factor, params = self.get_param('factor', 1, **params)
            n_keyphrases = int(n_keyphrases * factor)

        return extractor.get_n_best(n=n_keyphrases, redundancy_removal=redundancy_removal, stemming=(normalization == 'stemming')), extractor, params

    def _evaluate_document(self, model, input_document, references, evaluators, print_document_scores=True, **kwargs):
        language = kwargs.get('language', 'en')
        normalization = kwargs.get('normalization', 'stemming')
        redundancy_removal = kwargs.get('redundancy_removal', False)

        # make sure to initialize the evaluator clean for every run!
        for key, evaluator_data in evaluators.items():
            evaluator = Evaluator()
            evaluator.compare_func = evaluator_data['eval_comp_func']
            evaluators[key]['evaluator'] = evaluator

        if isinstance(input_document, str):
            filename = os.path.splitext(os.path.basename(input_document))[0]
            extractor = model()
            extractor.load_document(input_document, language=language, normalization=normalization)
        elif isinstance(input_document, dict):
            filename = input_document['id']
            extractor = input_document['document']
        reference = references[filename]

        print("Processing File: %s" % filename)
        keyphrases, context, adjusted_params = self.extract_keyphrases(model, extractor, filename, **kwargs)
        # Filter out reference keyphrases that don't appear in the original text
        if kwargs.get('filter_reference_keyphrases', False) is True:
            reference = self._filter_reference_keyphrases(reference, context, kwargs.get('normalization', 'stemming'))

        doc_eval_data = {}
        if (len(keyphrases) > 0 and len(reference) > 0):
            for key, evaluator_data in evaluators.items():
                evaluator = evaluator_data['evaluator']

                evaluator.evaluate(reference, list(zip(*keyphrases))[0])
                if print_document_scores is True:
                    print("%s - Precision: %s, Recall: %s, F-Score: %s" % (
                    key, evaluator.precision, evaluator.recall, evaluator.f_measure))

                doc_eval_data[key] = {
                    'precision': evaluator.precision,
                    'recall': evaluator.recall,
                    'f-score': evaluator.f_measure
                }

                evaluators[key]['precision_total'] += evaluator.precision
                evaluators[key]['recall_total'] += evaluator.recall
                evaluators[key]['f_score_total'] += evaluator.f_measure
        else:
            print(
                "Skipping file %s for not enough reference keyphrases or found keyphrases. Found keyphrases: %s, Reference keyphrases: %s" % (
                filename, len(keyphrases), len(reference)))

            for key, evaluator_data in evaluators.items():
                doc_eval_data[key] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f-score': 0.0
                }

        if adjusted_params.get('write_to_db', False) is True:
            database_handler = DatabaseHandler()
            if model is KeyCluster:
                database_handler.write_data_to_db(filename, doc_eval_data, data_cluster_members=extractor.data_cluster_members, data_candidate_keyphrases=extractor.data_candidate_keyphrases, **adjusted_params)
            elif model is TfIdf:
                unstemmed_keyphrases = extractor.get_n_best(n=adjusted_params.get('n_keyphrases', 10),
                                                            redundancy_removal=True,
                                                            stemming=False)
                database_handler.write_data_to_db(filename, doc_eval_data, data_candidate_keyphrases=unstemmed_keyphrases, **adjusted_params)
            else:
                unstemmed_keyphrases = extractor.get_n_best(n=adjusted_params.get('n_keyphrases', 10),
                                                            redundancy_removal=redundancy_removal, stemming=normalization)
                data_candidate_keyphrases = collect_keyphrase_data(extractor, unstemmed_keyphrases)
                database_handler.write_data_to_db(filename, doc_eval_data, data_candidate_keyphrases=data_candidate_keyphrases, **adjusted_params)
        return evaluators


    def calculate_model_f_score(self, model, input_data=None, references=None, print_document_scores=True, **kwargs):
        num_documents_evaluated = 0

        eval_comp_func_list = kwargs.get('evaluator_compare_func', [stemmed_compare])
        evaluators = dict()
        for eval_comp_func in eval_comp_func_list:
            evaluators[str(eval_comp_func.__name__)] = {
                'precision_total': 0,
                'recall_total': 0,
                'f_score_total': 0,
                'eval_comp_func': eval_comp_func
            }

        # Parse the frequent word list once for the model
        kwargs = _load_frequent_word_list(**kwargs)

        # Load a spacy model once for the model
        kwargs = _load_word_embedding_model(**kwargs)

        # Load the global cooccurrence matrix if specified
        kwargs = load_global_cooccurrence_matrix(**kwargs)

        # Load the document similarity data if specified
        kwargs = load_document_similarity_data(input_data, **kwargs)

        # Load the global covariance matrix if specified
        kwargs = load_global_covariance_matrix(input_data, **kwargs)

        # Load the sent2vec model
        kwargs.update({'sent2vec_model': EmbeddingDistributor(kwargs.get('sent2vec_model', '../word_embedding_models/english/sent2vec/wiki_bigrams.bin'))})

        # if input_data is None or references is None:
        if references is None and input_data is not None:
            db_handler = DatabaseHandler()

            print("Loading documents from the database for the %s dataset." % input_data)
            documents, references = db_handler.load_split_from_db(model, dataset=input_data, **kwargs)
            for key, doc in documents.items():
                self._evaluate_document(model, doc, references, evaluators, print_document_scores=print_document_scores,
                                        **kwargs)
                num_documents_evaluated += 1
                self._calc_avg_scores(evaluators, num_documents_evaluated, print_document_scores=print_document_scores)
        elif references is not None and input_data is not None:
            for file in glob.glob(input_data + '/*'):
                evaluators = self._evaluate_document(model, file, references, evaluators,
                                                     print_document_scores=print_document_scores, **kwargs)
                num_documents_evaluated += 1
                self._calc_avg_scores(evaluators, num_documents_evaluated, print_document_scores=print_document_scores)
        else:
            raise Exception("An error occured while trying to load the data. Make sure input_data and references was set "
                            "accordingly when loading a dataset from disk or that input_data was set to the name of the"
                            "corresponding dataset if the data is to be loaded from the database.")

        return self._calc_avg_scores(evaluators, num_documents_evaluated, print_document_scores=False)

    def extract_keyphrases_from_raw_text(self, model, input_document, input_data=None, **kwargs):
        # Parse the frequent word list once for the model
        kwargs = _load_frequent_word_list(**kwargs)

        # Load a spacy model once for the model
        kwargs = _load_word_embedding_model(**kwargs)

        # Load the global cooccurrence matrix if specified
        kwargs = load_global_cooccurrence_matrix(**kwargs)

        # Load the document similarity data if specified
        kwargs = load_document_similarity_data(input_data, **kwargs)

        # Load the global covariance matrix if specified
        kwargs = load_global_covariance_matrix(input_data, **kwargs)

        # Load the sent2vec model
        kwargs.update({'sent2vec_model': EmbeddingDistributor(kwargs.get('sent2vec_model', '../word_embedding_models/german/sent2vec/de_model.bin'))})

        language = kwargs.get('language', 'en')
        normalization = kwargs.get('normalization', 'stemming')

        filename = os.path.splitext(os.path.basename(input_document))[0]
        extractor = model()
        extractor.load_document(input_document, language=language, normalization=normalization)

        # Add ids to the sentences
        cur_id = 1
        for sent in extractor.sentences:
            if 'id' not in sent.meta.keys():
                sent.meta['id'] = cur_id
                cur_id += 1

        print("Processing File: %s" % filename)
        keyphrases, context, adjusted_params = self.extract_keyphrases(model, extractor, filename, **kwargs)

        # If stemming was used also retrieve the unstemmed keyphrases
        if normalization == 'stemming':
            unstemmed_keyphrases = extractor.get_n_best(n=len(keyphrases), redundancy_removal=False, stemming=False)
            return keyphrases, unstemmed_keyphrases
        else:
            return keyphrases, []

    def _calc_avg_scores(self, evaluators, num_documents, print_document_scores=True):
        for key, evaluator_data in evaluators.items():
            evaluators[key]['macro_precision'] = (evaluator_data['precision_total'] / num_documents)
            evaluators[key]['macro_recall'] = (evaluator_data['recall_total'] / num_documents)
            evaluators[key]['macro_f_score'] = (evaluator_data['f_score_total'] / num_documents)

            if print_document_scores is True:
                print("%s - Current Macro average precision: %s, recall: %s, f-score: %s" % (
                    key, evaluators[key]['macro_precision'], evaluators[key]['macro_recall'],
                    evaluators[key]['macro_f_score']))

        return evaluators

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
