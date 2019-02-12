"""
 This class was written by Johannes Villmow.
"""

from typing import Callable, NamedTuple, Sequence, Set
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

__stemmer = PorterStemmer()
ComparisonResult = NamedTuple('ComparisonResult', [('tp', int), ('fp', int), ('tn', int), ('fn', int)])
ValidationResult = NamedTuple('ValidationResult',
                              [('precision', float), ('recall', float), ('f_measure', float), ('error', float)])
IoUResult = NamedTuple('IoUResult', [('precision', float), ('recall', float)])

def error_rate(tp, tn, fp, fn) -> float:
    """
    Returns the error rate of a prediction.
    
    :param int tp: True positives
    :param int tn: True negatives
    :param int fp: False positives
    :param int fn: False negatives 

    :return: Error rate 
    :rtype: float
    """
    errors = float(fp + fn)
    all_ = (fp + fn + tp + tn)
    return errors / all_


def accuracy(tp, tn, fp, fn) -> float:
    """
    Returns the accuracy of a prediction.

    :param int tp: True positives
    :param int tn: True negatives
    :param int fp: False positives
    :param int fn: False negatives 
    
    :return: accuracy
    :rtype: float
    """

    return 1 - error_rate(tp, tn, fp, fn)


def false_positive_rate(fp, tn) -> float:
    """
    
    :param int fp: False positives
    :param int tn: True negatives
    :return: False positive rate
    :rtype: float
    """
    return float(fp) / (fp + tn)


def false_negative_rate(fn, tp) -> float:
    """

    :param int fn: False negatives
    :param int tp: True positives

    :return: False negative rate
    :rtype: float
    """
    return float(fn) / (fn + tp)


def true_positive_rate(tp, fn) -> float:
    """
    Recall is the number of True Positives divided by the number of True Positives and the number of False 
    Negatives. It is the number of positive predictions divided by the number of positive class values in the test 
    data. It is also called Sensitivity or the True Positive Rate .
    
    :param int tp: True positives
    :param int fn: False negatives 
    :return: True positive rate
    :rtype: float
    """
    if tp == 0:
        return 0
    else:
        return float(tp) / (tp + fn)


def true_negative_rate(tn, fp) -> float:
    """
    
    :param int tn: True negatives
    :param int fp: False positives

    :return: True negative rate
    :rtype: float
    """
    if tn == 0:
        return 0
    else:
        return float(tn) / (tn + fp)


def precision(tp, fp) -> float:
    """
    Precision is the number of True Positives divided by the number of True Positives and False Positives. 

    :param int tp: True positives
    :param int fp: False positives
    :return: Precision
    :rtype: float
    """
    if tp == 0:
        return 0
    else:
        return float(tp) / (tp + fp)


def recall(tp, fn) -> float:
    """
    Recall is the number of True Positives divided by the number of True Positives and the number of False 
    Negatives. It is the number of positive predictions divided by the number of positive class values in the test 
    data. It is also called Sensitivity or the True Positive Rate .

    :param int tp: True positives
    :param int fn: False negatives 
    :return: recall
    :rtype: float
    """
    return true_positive_rate(tp, fn)


def f_measure(f_precision: float, f_recall: float, beta=1) -> float:
    """
    The F-measure is the harmonic mean of precision and recall.
    One can extend the F-measure with beta to emphasize precision or recall: The higher beta, 
    the more important is recall (beta = 1 → recall and precision are equally important)

    :param f_precision: precision
    :param f_recall: recall
    :param beta: emphasize precision or recall
    :return: f measure
    :rtype: float
    """
    if f_precision == 0 or recall == 0:
        return 0
    else:
        return (1 + beta ** 2) * (float(f_precision * f_recall) /
                                  ((beta ** 2) * f_precision + f_recall))


def strict_compare(original_keywords, found_keywords) -> ComparisonResult:
    return _compare(original_keywords, found_keywords, _phrases)


def stemmed_compare(original_keywords, found_keywords) -> ComparisonResult:
    return _compare(original_keywords, found_keywords, _stemmed_phrases)


def word_compare(original_keywords: Sequence[str], found_keywords: Sequence[str]) -> ComparisonResult:
    return _compare(original_keywords, found_keywords, _phrases)


def stemmed_word_compare(original_keywords: Sequence[str], found_keywords: Sequence[str]) -> ComparisonResult:
    return _compare(original_keywords, found_keywords, _stemmed_words)


def stemmed_wordwise_phrase_compare(original_keywords, found_keywords) -> IoUResult:
    return _wordwise_compare(original_keywords, found_keywords, _stemmed_phrases)


def wordwise_phrase_compare(original_keywords, found_keywords) -> IoUResult:
    return _wordwise_compare(original_keywords, found_keywords, _phrases)


def _phrases(keyphrases: Sequence[str]) -> Set[str]:
    return {k.lower().strip() for k in keyphrases}


def _stemmed_phrases(keyphrases: Sequence[str]) -> Set[str]:
    return {_stem(phrase) for phrase in _phrases(keyphrases)}


def _words(keyphrases: Sequence[str]) -> Set[str]:
    return {word for phrase in _phrases(keyphrases) for word in _tokenize(phrase)}


def _stemmed_words(keyphrases: Sequence[str]) -> Set[str]:
    return {_stem(word) for word in _words(keyphrases)}


def _tokenize(phrase: str) -> Sequence[str]:
    return word_tokenize(phrase)


def _stem(word: str) -> str:
    return __stemmer.stem(word)


def _compare(original_keywords: Sequence[str], found_keywords: Sequence[str],
             transform: Callable[[Sequence], Set] = lambda x: set(x)) -> ComparisonResult:
    original_keywords = transform(original_keywords)
    found_keywords = transform(found_keywords)

    true_positives = original_keywords & found_keywords
    true_negatives = {}  # TODO calculate true_negatives ... this would be every word of the text, that is not a keyword
    false_positives = found_keywords - original_keywords
    false_negatives = original_keywords - found_keywords

    tp = len(true_positives)
    fp = len(false_positives)
    tn = len(true_negatives)
    fn = len(false_negatives)

    return ComparisonResult(tp, fp, tn, fn)


def _calculate_keyphrase_iou(keywords1, keywords2):
    scores = {}
    for t_ in keywords1:
        phrase_scores = []
        for t in keywords2:
            overlap = len(set(t_.split(' ')) & set(t.split(' ')))
            union = len(set(t_.split(' '))) + len(set(t.split(' '))) - overlap
            iou = overlap / union
            phrase_scores.append(iou)
        scores[t_] = max(phrase_scores)

    return scores


def _wordwise_compare(original_keywords: Sequence[str], found_keywords: Sequence[str],
             transform: Callable[[Sequence], Set] = lambda x: set(x)) -> IoUResult:
    original_keywords = transform(original_keywords)
    found_keywords = transform(found_keywords)
        
    found_scores = _calculate_keyphrase_iou(found_keywords, original_keywords)
    original_scores = _calculate_keyphrase_iou(original_keywords, found_keywords)

    rec = np.mean(list(original_scores.values()))
    prec = np.mean(list(found_scores.values()))

    return IoUResult(prec, rec)


class Evaluator:
    def __init__(self, compare_func: Callable = strict_compare, f_measure_beta: float = 1.0):
        """
        
        :param compare_func:    A function that takes a list of original and found keywords and 
                                returns precision, recall, f_measure and error_rate in that order
        """

        self.compare_func = compare_func
        self.f_measure_beta = f_measure_beta
        self.original_keyword_count = 0
        self.extracted_keyword_count = 0
        self.correct_keyword_count = 0

        self._sum_precision = 0.0
        self._sum_recall = 0.0
        self._sum_f_measure = 0.0
        self._sum_error_rate = 0.0
        self._count_evaluate = 0

    @property
    def precision(self) -> float:
        return self._sum_precision / self._count_evaluate

    @property
    def recall(self) -> float:
        return self._sum_recall / self._count_evaluate

    @property
    def f_measure(self) -> float:
        return f_measure(self.precision, self.recall, beta=1)

    @property
    def error_rate(self) -> float:
        return 0
        # return self._sum_error_rate / self._count_evaluate

    def reset(self) -> None:
        self._sum_precision = 0.0
        self._sum_recall = 0.0
        self._sum_f_measure = 0.0
        self._sum_error_rate = 0.0
        self._count_evaluate = 0

    def evaluate(self, original_keywords, found_keywords) -> ValidationResult:
        """
        Mit dieser Funktion können die gefundenen Keywords innerhalb eines Datums (Textes) ausgewertet werden. Die 
        Funktion gibt die, für dieses Datum, ermittelten Statistiken zurück. Ein Evaluator Objekt speichert die Werte 
        zudem aggregiert als Attribut, so dass Aussagen über den gesamten Datensatz getroffen werden können.        
        
        :param list original_keywords: List of golden standard keywords
        :param list found_keywords: List of found keywords

        :return: precision, recall, f_measure, error_rate
        """

        comp_res = self.compare_func(original_keywords, found_keywords)
        if isinstance(comp_res, ComparisonResult):
            tp, fp, _, fn = comp_res

            if tp == 0 and fn == 0:
                # Corner Case:
                prec = 1.0
                rec = 1.0
            else:
                prec = precision(tp, fp)
                rec = recall(tp, fn)
            fmeasure = 0.0
        elif isinstance(comp_res, IoUResult):
            prec, rec = comp_res

        try:
            fmeasure = f_measure(f_precision=prec, f_recall=rec, beta=1)
        except ZeroDivisionError:
            pass

        # error rate = error_rate(tp, tn, fp, fn)
        errorrate = 0

        self._sum_precision += prec
        self._sum_recall += rec
        self._sum_error_rate += errorrate

        self.extracted_keyword_count += len(found_keywords)
        self.original_keyword_count += len(original_keywords)
        # self.correct_keyword_count += tp

        self._count_evaluate += 1

        return ValidationResult(prec, rec, fmeasure, errorrate)


def main():
    e = Evaluator()
    e.compare_func = stemmed_wordwise_phrase_compare #stemmed_compare

    # for i in range(10):
    print(e.evaluate(["test1 test2", "test1 test4"], ["test1 test2", "test3 test4", "test5"]))

    print(e.precision)
    print(e.recall)
    print(e.f_measure)
    print(e.error_rate)


if __name__ == '__main__':
    main()
