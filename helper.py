from string import punctuation
from pke import compute_document_frequency, compute_lda_model
from nltk.stem.snowball import SnowballStemmer
import pke
from nltk.tag.mapping import map_tag
import logging
import spacy
import numpy as np

_SPACY_MODELS = {
    'en_vectors_web_lg',
    'de_core_news_sm'
}


def reformat_glove_model(gloveFile):
    """
    Reformats a txt file into the word2vec file.
    The input file needs to be a txt file where each line consists of the word followed by its word embedding vector.
    To generate a scipy model out of this use:  python -m spacy init-model en output_dir --vectors-loc input_file

    :param gloveFile:
    :return:
    """
    words = 0
    dimension = 0
    recoded_file = ""
    print("Reformatting Glove Model")
    f = open(gloveFile, 'r')

    for line in f:
        splitLine = line.split()
        if words == 0:
            embedding = np.array([float(val) for val in splitLine[1:]])
            dimension = len(embedding)
        recoded_file += line
        words += 1
    recoded_file = str(words) + " " + str(dimension) + "\n" + recoded_file
    with open(gloveFile[:-3] + "recoded.txt", 'w') as f:
        f.write(recoded_file)


def _load_frequent_word_list(**kwargs):
    frequent_word_list = kwargs.get('frequent_word_list', None)
    if frequent_word_list is not None:
        print("frequent_word_list was set, ignoring frequent_word_list_file")
        return kwargs

    frequent_word_list_path = kwargs.get('frequent_word_list_file', '')
    if isinstance(frequent_word_list_path, str):
        language = kwargs.get('language', 'en')
        normalization = kwargs.get('normalization', 'stemming')
        if frequent_word_list_path.split('/')[-1].startswith('en_') and language == 'de':
            logging.warning("The language is set to german while possibly using a english frequent word list. Make "
                            "sure you have set frequent_word_list and language to the correct values.")
        elif frequent_word_list_path.split('/')[-1].startswith('de_') and language == 'en':
            logging.warning("The language is set to english while possibly using a german frequent word list. Make "
                            "sure you have set frequent_word_list and language to the correct values.")

        min_word_count = kwargs.get('min_word_count', 10000)
        frequent_word_list = parse_frequent_word_list(frequent_word_list_path, min_word_count=min_word_count,
                                                      language=language,
                                                      stemming=(normalization == 'stemming'))
        kwargs['frequent_word_list'] = frequent_word_list

    return kwargs


def _load_word_embedding_model(**kwargs):
    word_embedding_model = kwargs.get('word_embedding_model', None)
    if word_embedding_model is not None:
        print("word_embedding_model was set, ignoring word_embedding_model_file")
        return kwargs

    word_embedding_model_path = kwargs.get('word_embedding_model_file', 'en_vectors_web_lg')
    kwargs['word_embedding_model'] = spacy.load(word_embedding_model_path)
    if word_embedding_model_path in _SPACY_MODELS:
        print("Finished loading spacy model: %s" % word_embedding_model_path)
    else:
        print("Finished loading custom spacy model: %s" % word_embedding_model_path)
    return kwargs


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


def compute_df(input_dir, output_file, extension="xml"):
    stoplist = list(punctuation)
    compute_document_frequency(input_dir=input_dir,
                               output_file=output_file,
                               extension=extension,           # input file extension
                               language='en',                # language of files
                               normalization="stemming",    # use porter stemmer
                               stoplist=stoplist)


def parse_frequent_word_list(path, min_word_count=10000, language='en', stemming=False):
    if language == 'en':
        stemmer = SnowballStemmer("porter")
    else:
        stemmer = SnowballStemmer(pke.base.ISO_to_language[language], ignore_stopwords=False)

    if path == '':
        return []

    with open(path, encoding="utf-8") as f:
        content = f.readlines()
        content = dict(x.strip().split(' ') for x in content)

        if stemming is True:
            frequent_word_dict = dict()
            for word, count in sorted(content.items(), key=lambda x: int(x[1]), reverse=True):
                count = int(count)
                if count < min_word_count:
                    break

                word = stemmer.stem(word)
                if word not in frequent_word_dict.keys():
                    frequent_word_dict[word] = count
        else:
            frequent_word_dict = {k: v for k, v in content.items() if int(v) >= min_word_count}

    return list(frequent_word_dict.keys())


def calc_num_cluster(**params):
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