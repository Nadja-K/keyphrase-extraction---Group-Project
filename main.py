import pke
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
from ClusterFeatureCalculator import CooccurrenceClusterFeature
from CandidateTermSelector import CandidateTermSelector
from Cluster import HierarchicalClustering
from KeyphraseSelector import KeyphraseSelector
from evaluation import Evaluator

from nltk.tag.mapping import map_tag


def custom_normalize_POS_tags(self):
    """Normalizes the PoS tags from udp-penn to UD."""

    if self.language == 'en':
        # iterate throughout the sentences
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].pos = [map_tag('en-ptb', 'universal', tag) for tag in sentence.pos]
    elif self.language == 'de':
        # iterate throughout the sentences
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].pos = [map_tag('de-tiger', 'universal', tag) for tag in sentence.pos]


class KeyphraseExtractor:
    def num_cluster(self, **params):
        """
        Default method to calculate the number of clusters for cluster-based methods.

        :param int num_clusters
        :param float factor
        :param LoadFile context

        :return: int
        """
        num_clusters = params.get('num_clusters', 0)
        if num_clusters > 0:
            return num_clusters

        factor = params.get('factor', 1)
        context = params.get('context', None)
        if context is None:
            return 10

        return int(factor * len(context.candidate_terms))

    def extract_keyphrases(self, model, file, **params):
        language = params.get('language', 'en')
        normalization = params.get('normalization', 'stemming')
        frequency_file = params.get('frequency_file', None)
        redundancy_removal = params.get('redundancy_removal', False)

        extractor = model()
        extractor.load_document(file, language=language, normalization=normalization)

        df = None
        if frequency_file is not None:
            df = pke.load_document_frequency_file(input_file=frequency_file, encoding="utf-8")

        if model in [TfIdf]:
            """
            :param list stoplist
            :param int n_grams
            :param str frequency_file
            """
            stoplist = params.get('stoplist', extractor.stoplist)
            n_grams = params.get('n_grams', 3)

            extractor.candidate_selection(n=n_grams, stoplist=stoplist)
            extractor.candidate_weighting(df=df, encoding="utf-8")

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
            top_percent = params.get('top_percent', None)
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
            alpha = params.get('alpha', 1.1)

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
            extractor.candidate_weighting(window=10, pos=pos, lda_model=lda_model, stoplist=stoplist, normalized=normalized)

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
            :param CandidateTermSelector candidate_term_selector
            :param ClusterFeatureCalculator cluster_feature_calculator
            :param Cluster cluster_method
            :param KeyphraseSelector keyphrase_selector
            :param int window
            :param func cluster_calc
            :param dict cluster_calc_args
            FIXME
            """
            # FIXME
            window = params.get('window', 2)
            candidate_term_selector = params.get('candidate_term_selector', CandidateTermSelector())
            cluster_feature_calculator = params.get('cluster_feature_calculator', CooccurrenceClusterFeature(window=window))
            cluster_method = params.get('cluster_method', HierarchicalClustering())
            keyphrase_selector = params.get('keyphrase_selector', KeyphraseSelector())

            extractor.candidate_selection(candidate_term_selector=candidate_term_selector)

            cluster_calc = params.get('cluster_calc', self.num_cluster)
            cluster_calc_args = params.get('cluster_calc_args', {'factor': 2/3})
            cluster_calc_args['context'] = extractor
            num_clusters = cluster_calc(**cluster_calc_args)

            extractor.candidate_weighting(cluster_feature_calculator=cluster_feature_calculator,
                                          cluster_method=cluster_method,
                                          keyphrase_selector=keyphrase_selector,
                                          num_clusters=num_clusters)

        else:
            extractor.candidate_selection()
            extractor.candidate_weighting()

        n_keyphrases = params.get('n_keyphrases', len(extractor.candidates))
        return extractor.get_n_best(n=n_keyphrases, redundancy_removal=redundancy_removal, stemming=(normalization == 'stemming'))

    def calculate_model_f_score(self, model, input_dir, references, **kwargs):
        precision_total = 0
        recall_total = 0
        f_score_total = 0
        num_documents = 0

        for file in glob.glob(input_dir + '/*'):
            # make sure to initialize the evaluator clean for every run!
            evaluator = Evaluator()

            filename = os.path.splitext(os.path.basename(file))[0]
            reference = references[filename]

            keyphrases = self.extract_keyphrases(model, file, **kwargs)
            evaluator.evaluate(list(zip(*keyphrases))[0], reference)
            # print(list(zip(*keyphrases))[0])
            # print(reference)
            print("Precision: %s, Recall: %s, F-Score: %s" % (evaluator.precision, evaluator.recall, evaluator.f_measure))

            precision_total += evaluator.precision
            recall_total += evaluator.recall
            f_score_total += evaluator.f_measure
            num_documents += 1

        macro_precision = (precision_total / num_documents)
        macro_recall = (recall_total / num_documents)
        macro_f_score = (f_score_total / num_documents)
        return macro_precision, macro_recall, macro_f_score


kwargs = {
    'language': 'en',                   # load_document
    'normalization': 'stemming',        # load_document, YAKE, get_n_best
    'n_keyphrases': 10,                 # get_n_best
    # 'redundancy_removal': ,           # get_n_best
    # 'n_grams': ,                      # TfIdf, YAKE, #FIXME#
    # 'stoplist': ,                     # TfIdf, TopicRank, MultipartiteRank, TopicalPageRank, YAKE, KPMiner, #FIXME#
    # 'frequency_file': ,               # load_document, (TfIdf), (KPMiner)
    # 'window': ,                       # TextRank, SingleRank, TopicalPageRank, PositionRank, YAKE, KeyCluster
    # 'pos': ,
    # 'top_percent': ,
    # 'normalized': ,
    # 'run_candidate_selection': ,
    # 'threshold': ,
    # 'method': ,
    # 'heuristic': ,
    # 'alpha': ,
    # 'grammar': ,
    # 'lda_model': ,
    # 'normalized': ,
    # 'maximum_word_number': ,
    # 'lasf': ,
    # 'cutoff': ,
    # 'sigma': ,
    # 'candidate_term_selector': ,
    # 'cluster_feature_calculator': ,
    # 'cluster_method': ,
    # 'keyphrase_selector': ,
    # 'cluster_calc': ,
    'cluster_calc_args': {'num_clusters': 10}
}


def custom_testing():
    inspec_test_folder = "../ake-datasets/datasets/Inspec/test"
    inspec_controlled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.contr.stem.json"
    inspec_uncontrolled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json"
    inspec_controlled_stemmed = pke.utils.load_references(inspec_controlled_stemmed_file)
    inspec_uncontrolled_stemmed = pke.utils.load_references(inspec_uncontrolled_stemmed_file)

    heise_folder = "../ake-datasets/datasets/Heise"

    extractor = KeyphraseExtractor()
    models = [
        KeyCluster,
        # TfIdf,
        # TopicRank,
        # SingleRank,
        # KPMiner
    ]

    for m in models:
        print("Computing the F-Score for the Inspec Dataset with {}".format(m))
        macro_precision, macro_recall, macro_f_score = extractor.calculate_model_f_score(m, inspec_test_folder, inspec_uncontrolled_stemmed, **kwargs)
        print("Macro average precision: %s, recall: %s, f-score: %s" % (macro_precision, macro_recall, macro_f_score))

    # i = 0
    # for file in glob.glob(heise_folder + '/*'):
    #     if i >= 0:
    #         # filename = os.path.splitext(os.path.basename(file))[0]
    #         # reference = inspec_uncontrolled_stemmed[filename]
    #         keyphrases = extractor.extract_keyphrases(m, file, **kwargs)
    #         print(list(zip(*keyphrases))[0])
    #         print()
    #     i += 1
    #     if i == 2:
    #         break


if __name__ == '__main__':
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    custom_testing()
