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
from KeyCluster import KeyCluster

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


def calculate_f_score(references, extracted):
    # print("\nUncontrolled unstemmed reference keyphrases: %s" % references)
    # print("Extracted keyphrases: %s" % extracted)

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
        extractor.candidate_weighting(window=2, pos=pos, top_percent=1.0) # 0.33)

    elif model in [SingleRank]:
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(window=10, pos=pos)

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

    else:
        extractor.candidate_selection()
        extractor.candidate_weighting()

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
    # reference values http://aclweb.org/anthology/C16-2015

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
        # TfIdf,          # 15.2% vs 16.4%
        # TopicRank,      # 12.6 vs 12.6%
        # SingleRank,     # 1.9% vs 1.8%
        KPMiner         # 19.4% vs 19.8%
    ]
    for m in models:
        print("Computing the F-Score for the SemEval-2010 Dataset with {}".format(m))
        precision, recall, micro_f_score, macro_f_score = calculate_model_f_score(m, semeval_test_folder, semeval_combined_stemmed, '../ake-datasets/datasets/SemEval-2010/SemEval_df_counts.tsv.gz', '../ake-datasets/datasets/SemEval-2010/lda_model.tsv.gz')
        print("Micro average precision: %s, recall: %s, f_score: %s" % (precision, recall, micro_f_score))  # <-- this is used in the SemEval-2010 competition https://www.aclweb.org/anthology/S10-1004
        print("Macro average f-score: %s" % macro_f_score)


def inspec_testing():
    # reference values for the uncontrolled keyphrases: https://arxiv.org/pdf/1801.04470.pdf
    # Anm.: Die Werte im Paper scheinen auf den uncontrolled keyphrases zu basieren

    inspec_test_folder = "../ake-datasets/datasets/Inspec/test"
    inspec_controlled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.contr.stem.json"
    inspec_uncontrolled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json"
    inspec_controlled_stemmed = pke.utils.load_references(inspec_controlled_stemmed_file)
    inspec_uncontrolled_stemmed = pke.utils.load_references(inspec_uncontrolled_stemmed_file)

    # compute_df('../ake-datasets/datasets/Inspec/train',
    #            '../ake-datasets/datasets/Inspec/Inspec_df_counts.tsv.gz', extension="xml")
    # compute_lda_model('../ake-datasets/datasets/Inspec/train',
    #                   '../ake-datasets/datasets/Inspec/lda_model.tsv.gz', extension="xml")
    models = [
                            # own macro f-score vs. paper macro f-score
        TextRank,         # 34.23%/14.88% vs 15.28% with 1.0-top/0.33-top
        # SingleRank,       # 34.44% vs 36.51%
        # TopicRank,        # 28.42% vs 29.02%
        # MultipartiteRank  # 29.29% vs 30.01%
    ]
    for m in models:
        print("Computing the F-Score for the Inspec Dataset with {}".format(m))
        # precision, recall, micro_f_score, macro_f_score = calculate_model_f_score(m, inspec_test_folder,
        #                                                                           inspec_controlled_stemmed,
        #                                                                           '../ake-datasets/datasets/Inspec/Inspec_df_counts.tsv.gz',
        #                                                                           '../ake-datasets/datasets/Inspec/lda_model.tsv.gz')
        precision, recall, micro_f_score, macro_f_score = calculate_model_f_score(m, inspec_test_folder,
                                                                                  inspec_uncontrolled_stemmed,
                                                                                  '../ake-datasets/datasets/Inspec/Inspec_df_counts.tsv.gz',
                                                                                  '../ake-datasets/datasets/Inspec/lda_model.tsv.gz')
        print("Micro average precision: %s, recall: %s, f_score: %s" % (precision, recall, micro_f_score))  # <-- this is used in the SemEval-2010 competition https://www.aclweb.org/anthology/S10-1004
        print("Macro average f-score: %s" % macro_f_score)  # <-- this is used in the paper mentioned above for the reference values


def duc_testing():
    # reference values for the uncontrolled keyphrases: https://arxiv.org/pdf/1801.04470.pdf

    duc_test_folder = "../ake-datasets/datasets/DUC-2001/test"
    duc_stemmed_file = "../ake-datasets/datasets/DUC-2001/references/test.reader.stem.json"
    duc_stemmed = pke.utils.load_references(duc_stemmed_file)

    models = [
                            # own macro f-score vs. paper macro f-score
        TextRank,         #  17.43%/12.35% vs 15.24% with 1.0-top/0.33-top
        # SingleRank,       # 24.97% vs 27.51%
        # TopicRank,        # 22.89% vs 24.04%
        # MultipartiteRank  # 25.06% vs 25.28%
    ]
    for m in models:
        print("Computing the F-Score for the DUC-2001 Dataset with {}".format(m))
        precision, recall, micro_f_score, macro_f_score = calculate_model_f_score(m, duc_test_folder,
                                                                                  duc_stemmed,
                                                                                  None, None)
        print("Micro average precision: %s, recall: %s, f_score: %s" % (precision, recall, micro_f_score))  # <-- this is used in the SemEval-2010 competition https://www.aclweb.org/anthology/S10-1004
        print("Macro average f-score: %s" % macro_f_score)  # <-- this is used in the paper mentioned above for the reference values


def nus_testing():
    # reference values for the uncontrolled keyphrases: https://arxiv.org/pdf/1801.04470.pdf

    nus_test_folder = "../ake-datasets/datasets/NUS/test"
    nus_stemmed_file = "../ake-datasets/datasets/NUS/references/test.combined.stem.json" # <-- this one is used in the paper mentioned above
    nus_stemmed = pke.utils.load_references(nus_stemmed_file)

    models = [
                            # own macro f-score vs. paper macro f-score
        # TextRank,         # 0.68%/1.77% vs 6.56% with 1.0-top/0.33-top
        # SingleRank,       # 2.3% vs 5.13%
        # TopicRank,        # 15.21% vs 13.81%
        MultipartiteRank  # 18.01% vs 16.92%
    ]
    for m in models:
        print("Computing the F-Score for the NUS Dataset with {}".format(m))
        precision, recall, micro_f_score, macro_f_score = calculate_model_f_score(m, nus_test_folder,
                                                                                  nus_stemmed,
                                                                                  None, None)
        print("Micro average precision: %s, recall: %s, f_score: %s" % (precision, recall, micro_f_score))  # <-- this is used in the SemEval-2010 competition https://www.aclweb.org/anthology/S10-1004
        print("Macro average f-score: %s" % macro_f_score)  # <-- this is used in the paper mentioned above for the reference values


def custom_testing():
    inspec_test_folder = "../ake-datasets/datasets/Inspec/test"
    inspec_controlled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.contr.stem.json"
    inspec_uncontrolled_stemmed_file = "../ake-datasets/datasets/Inspec/references/test.uncontr.stem.json"
    inspec_controlled_stemmed = pke.utils.load_references(inspec_controlled_stemmed_file)
    inspec_uncontrolled_stemmed = pke.utils.load_references(inspec_uncontrolled_stemmed_file)

    i = 0
    for file in glob.glob(inspec_test_folder + '/*'):
        if i >= 0:
            filename = os.path.splitext(os.path.basename(file))[0]
            print(file)
            reference = inspec_uncontrolled_stemmed[filename]
            keyphrases = extract_keyphrases(KeyCluster, file, normalization="stemming", n_keyphrases=30)
            print(keyphrases)
        i += 1
        if i == 10:
            break
if __name__ == '__main__':
    # inspec_testing()
    # semeval_testing()
    # duc_testing()
    # nus_testing()
    custom_testing()