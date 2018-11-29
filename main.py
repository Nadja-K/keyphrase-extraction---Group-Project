import json

import pke
from nltk.corpus import stopwords
from pke import compute_document_frequency
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

def compute_df(input_dir, output_file, extension="xml"):
    stoplist = list(punctuation)
    compute_document_frequency(input_dir=input_dir,
                               output_file=output_file,
                               extension=extension,           # input file extension
                               language='en',                # language of files
                               normalization="stemming",    # use porter stemmer
                               stoplist=stoplist)


def compute_lda_model(input_dir, output_file, extension="xml"):
    pke.utils.compute_lda_model(input_dir=input_dir,
                                  output_file=output_file,
                                  n_topics=500,
                                  extension=extension,
                                  language="en",
                                  normalization="stemming")



def topic_rank(input_path, normalization=None):
    extractor = pke.unsupervised.TopicRank()

    pos = {'NOUN', 'PROPN', 'ADJ'}
    stoplist = list(punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')

    extractor.load_document(input=input_path, normalization=normalization)
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    extractor.candidate_weighting(threshold=0.74, method='average')

    keyphrases = extractor.get_n_best(n=10, stemming=(normalization == "stemming"))

    return keyphrases


def topic_page_rank(input_path, lda_model_file):
    extractor = pke.unsupervised.TopicalPageRank()

    pos = {'NOUN', 'PROPN', 'ADJ'}
    grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

    extractor.load_document(input=input_path, language='en', normalization=None)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos, lda_model=lda_model_file)

    keyphrases = extractor.get_n_best(n=10)

    return keyphrases


def test_with_document():
    # universal test file
    input_doc = "..\\test_docs\\test.final"

    # stuff that needs to be done only once
    # compute_df('..\\test_docs', '..\\test_docs\\document_frequency_counts.tsv.gz', extension="final")
    # compute_lda_model('..\\test_docs', '..\\test_docs\\lda_model.tsv.gz', extension="final")

    # using various baselines
    # print(tfidf(input_doc, frequency_file='..\\test_docs\\document_frequency_counts.tsv.gz'))
    # print(topic_rank(input_doc))
    print(topic_page_rank(input_doc, '..\\test_docs\\lda_model.tsv.gz'))


def calculate_f_score(references, extracted):
    print("\nUncontrolled unstemmed reference keyphrases: %s" % references)
    print("Extracted keyphrases: %s" % extracted)

    true_positive = 0
    false_positive = 0
    for keyphrase, score in extracted:
        if keyphrase in references:
            true_positive += 1
        else:
            false_positive += 1

    precision = true_positive / len(extracted)
    recall = true_positive / len(references)
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = ((2 * precision * recall) / (precision + recall))

    print(precision, recall, f_score)
    return precision, recall, f_score, true_positive, false_positive


def extract_keyphrases(model, file, normalization=None, n_grams=3, n_keyphrases=10, frequency_file=None, lda_model=None):
    extractor = model()
    extractor.load_document(file, normalization=normalization)

    df = None
    pos = {'NOUN', 'PROPN', 'ADJ'}
    grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
    if frequency_file is not None:
        df = pke.load_document_frequency_file(input_file=frequency_file, encoding="utf-8")

    if model in [TfIdf]:
        extractor.candidate_selection(n=n_grams, stoplist=list(punctuation))
        extractor.candidate_weighting(df=df, encoding="utf-8")

    elif model in [TextRank]:
        extractor.candidate_weighting(window=2, pos=pos, top_percent=0.33)

    elif model in [SingleRank]:
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(window=2, pos=pos)

    elif model in [TopicRank, MultipartiteRank]:
        stoplist = list(punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_weighting()

    elif model in [TopicalPageRank]:
        extractor.candidate_selection(grammar=grammar)
        extractor.candidate_weighting(window=10, pos=pos, lda_model=lda_model)

    elif model in [PositionRank]:
        extractor.candidate_selection(grammar=grammar, maximum_word_number=3)
        extractor.candidate_weighting(window=10, pos=pos)

    elif model in [YAKE]:
        stoplist = stopwords.words('english')
        extractor.candidate_selection(n=n_grams, stoplist=stoplist)
        extractor.candidate_weighting(window=2, stoplist=stoplist, use_stems=(normalization == 'stemming'))

    elif model in [KPMiner]:
        extractor.candidate_selection(lasf=3, cutoff=400)
        extractor.candidate_weighting(df=df, alpha=2.3, sigma=3.0, encoding="utf-8")

    return extractor.get_n_best(n=n_keyphrases, stemming=(normalization == 'stemming'))


def calculate_model_f_score(model, input_dir, references, frequency_file=None, lda_model=None):
    true_positive_total = 0
    num_extracted_keyphrases = 0
    num_reference_keyphrases = 0
    f_score_total = 0
    num_documents = 0

    for file in glob.glob(input_dir + '/*'):
        filename = os.path.splitext(os.path.basename(file))[0]
        reference = references[filename]

        keyphrases = extract_keyphrases(model, file, normalization="stemming", frequency_file=frequency_file, lda_model=lda_model)
        precision, recall, f_score, true_positive, false_positive = calculate_f_score(reference, keyphrases)

        true_positive_total += true_positive
        num_reference_keyphrases += len(reference)
        num_extracted_keyphrases += len(keyphrases)
        f_score_total += f_score
        num_documents += 1

    precision = true_positive_total / num_extracted_keyphrases
    recall = true_positive_total / num_reference_keyphrases
    if precision == 0 or recall == 0:
        micro_f_score = 0
    else:
        micro_f_score = ((2 * precision * recall) / (precision + recall))

    macro_f_score = (f_score_total / num_documents)
    return precision, recall, micro_f_score, macro_f_score


def semeval_testing():
    semeval_test_folder = "../ake-datasets/datasets/SemEval-2010/test"
    semeval_combined_stemmed_file = "../ake-datasets/datasets/SemEval-2010/references/test.combined.stem.json"
    semeval_combined_stemmed = pke.utils.load_references(semeval_combined_stemmed_file)

    # this only needs to be done once for the train documents
    # compute_df('../ake-datasets/datasets/SemEval-2010/train', '../ake-datasets/datasets/SemEval-2010/SemEval_df_counts.tsv.gz', extension="xml")
    # compute_lda_model('../ake-datasets/datasets/SemEval-2010/train', '../ake-datasets/datasets/SemEval-2010/lda_model.tsv.gz', extension="xml")

    # models = [
    #     TfIdf, TopicRank, SingleRank, MultipartiteRank, PositionRank, TopicalPageRank, ExpandRank,
    #     TextRank, KPMiner, YAKE, FirstPhrases
    # ]
    models = [
        KPMiner
    ]
    for m in models:
        print("Computing the F-Score for the SemEval-2010 Dataset with {}".format(m))
        precision, recall, micro_f_score, macro_f_score = calculate_model_f_score(m, semeval_test_folder, semeval_combined_stemmed, '../ake-datasets/datasets/SemEval-2010/SemEval_df_counts.tsv.gz', '../ake-datasets/datasets/SemEval-2010/lda_model.tsv.gz')
        print("Micro average precision: %s, recall: %s, f_score: %s" % (precision, recall, micro_f_score))  # <-- this is used in the SemEval-2010 competition https://www.aclweb.org/anthology/S10-1004
        print("Macro average f-score: %s" % macro_f_score)


def inspec_testing():
    inspec_test_folder = "../ake-datasets/datasets/Inspec/test"

    inspec_controlled_file = "../ake-datasets/datasets/Inspec/references/test.contr.json"
    inspec_controlled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.contr.stem.json"

    with open(inspec_controlled_file) as f:
        inspec_controlled = json.load(f)

    with open(inspec_controlled_stemmed_file) as f:
        inspec_controlled_stemmed = json.load(f)

    true_positive_total = 0
    false_positive_total = 0
    false_negative_total = 0
    for file in glob.glob(inspec_test_folder + '/*'):
        filename = os.path.splitext(os.path.basename(file))[0]
        print("\nUncontrolled unstemmed reference keyphrases: %s" % (inspec_controlled[filename]))
        print("Extracted keyphrases: %s" % (tfidf(file, n_keyphrases=10)))

        keyphrases = tfidf(file, n_keyphrases=10)
        true_positive = 0
        false_positive = 0
        for keyphrase, score in keyphrases:
            if keyphrase in inspec_controlled[filename]:
                true_positive += 1
            else:
                false_positive += 1
        false_negative = len(inspec_controlled[filename]) - true_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = ((2 * precision * recall) / (precision + recall))

        true_positive_total += true_positive
        false_positive_total += false_positive
        false_negative_total += false_negative
        print(precision, recall, f_score)


    precision = true_positive_total / (true_positive_total + false_positive_total)
    recall = true_positive_total / (true_positive_total + false_negative_total)
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = ((2 * precision * recall) / (precision + recall))
    print("Total precision, recall and f_score")
    print(precision, recall, f_score)



if __name__ == '__main__':
    # inspec_testing()
    semeval_testing()