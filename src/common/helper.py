from collections import defaultdict
from string import punctuation

from pandas import json
from pke import compute_document_frequency, compute_lda_model, Candidate
from nltk.stem.snowball import SnowballStemmer
import pke
from nltk.tag.mapping import map_tag
import logging
import spacy
import numpy as np
import time
import glob
import string
import json

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import os
import gzip
import networkx as nx
import re

from common.ClusterFeatureCalculator import CooccurrenceClusterFeature, PPMIClusterFeature, WordEmbeddingsClusterFeature
from common.DatabaseHandler import DatabaseHandler
from methods.KeyCluster import KeyCluster

_SPACY_MODELS = {
    'en_vectors_web_lg',
    'de_core_news_sm'
}


def collect_keyphrase_data(context, selected_candidates):
    selected_candidates = list(zip(*selected_candidates))[0]
    text = []

    # Get the starting offset for each sentence
    sentence_start_offset = 0
    for sentence in context.sentences:
        raw_sentence = ' '.join(sentence.words)
        text.append([raw_sentence, sentence_start_offset])
        sentence_start_offset += len(raw_sentence) + 1

    data_canidate_keyphrases = dict()
    for term, candidate in context.candidates.items():
        data_canidate_keyphrases[term] = dict()
        data_canidate_keyphrases[term]['offsets'] = []
        data_canidate_keyphrases[term]['exemplar_terms_count'] = 0
        data_canidate_keyphrases[term]['pos'] = candidate.pos_patterns
        data_canidate_keyphrases[term]['selected'] = False
        data_canidate_keyphrases[term]['sentence_id'] = candidate.sentence_ids
        data_canidate_keyphrases[term]['stems'] = candidate.lexical_form
        data_canidate_keyphrases[term]['weight'] = context.weights[term].item()
        data_canidate_keyphrases[term]['words'] = candidate.surface_forms

        if term in selected_candidates:
            data_canidate_keyphrases[term]['selected'] = True

        # Extract the exact offsets for each candidate occurrence in the original text
        for sentence_id, surface_form in zip(candidate.sentence_ids, candidate.surface_forms):
            candidate_offsets = []

            raw_surface_form = ' '.join(surface_form)
            start_offsets = [m.start() for m in
                             re.finditer(r'\b(' + re.escape(raw_surface_form) + r')\b', text[sentence_id][0])]
            # Find all occurrences of the current candidate in the current sentence and add the offsets
            for start_offset in start_offsets:
                start_offset += text[sentence_id][1]
                end_offset = start_offset + len(raw_surface_form)
                candidate_offsets.append([start_offset, end_offset])

            if len(candidate_offsets) > 0:
                data_canidate_keyphrases[term]['offsets'].extend(candidate_offsets)

    return data_canidate_keyphrases


def reformat_glove_model(gloveFile):
    """
    Reformats a txt file into the word2vec file.
    The input file needs to be a txt file where each line consists of the word followed by its word embedding vector.
    To generate a scipy model out of this use:  python -m spacy init-model en output_dir --vectors-loc input_file

    To reformat a gensim model use the following 2 lines:
    trained_model = gensim.models.KeyedVectors.load_word2vec_format(input_model_path, binary=True)
    trained_model.save_word2vec_format(output_model_path + ".txt", binary=False)
    Then generate a scipy model with the above command.

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
            time.sleep(10)
        elif frequent_word_list_path.split('/')[-1].startswith('de_') and language == 'en':
            logging.warning("The language is set to english while possibly using a german frequent word list. Make "
                            "sure you have set frequent_word_list and language to the correct values.")
            time.sleep(10)

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

    if kwargs.get('cluster_feature_calculator', CooccurrenceClusterFeature) == WordEmbeddingsClusterFeature:
        word_embedding_model_path = kwargs.get('word_embedding_model_file', 'en_vectors_web_lg')

        kwargs['word_embedding_model_path'] = word_embedding_model_path
        kwargs['word_embedding_model'] = spacy.load(word_embedding_model_path)
        if word_embedding_model_path in _SPACY_MODELS:
            print("Finished loading spacy model: %s" % word_embedding_model_path)
        else:
            print("Finished loading custom spacy model: %s" % word_embedding_model_path)
    else:
        kwargs['word_embedding_model'] = None
        print("No word embedding model loaded.")
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


def compute_db_document_frequency(output_file, dataset='Heise', extension='xml', delimiter='\t', **kwargs):
    language = kwargs.get('language', 'en')
    normalization = kwargs.get('normalization', 'stemming')
    window = kwargs.get('window', 2)
    n_grams = kwargs.get('n_grams', 3)
    stoplist = kwargs.get('stoplist', None)
    batch_size = kwargs.get('batch_size', 100)
    num_documents = kwargs.get('num_documents', 0)
    frequencies = defaultdict(set)

    document_names = []

    print("Loading documents from the database for the %s dataset." % dataset)
    db_handler = DatabaseHandler()

    if num_documents == 0:
        num_documents = db_handler.get_num_documents_with_keyphrases(**kwargs)

    while (num_documents > 0):
        documents, _ = db_handler.load_documents_from_db(KeyCluster, **kwargs)
        for key, doc in documents.items():
            document_names.append(key)

            logging.info('reading file ' + key)

            doc['document'].ngram_selection(n=n_grams)
            doc['document'].candidate_filtering(stoplist=stoplist)
            for lexical_form in doc['document'].candidates:
                frequencies[lexical_form].add(key)

        num_documents -= batch_size
        print("Done with batch.")

    num_documents = len(document_names)

    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with gzip.open(output_file, 'wb') as f:
        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(num_documents)
        f.write(first_line.encode('utf-8') + b'\n')

        for ngram in frequencies:
            line = ngram + delimiter + str(len(frequencies[ngram]))
            f.write(line.encode('utf-8') + b'\n')


def compute_global_cooccurrence(output_file, input_dir=None, dataset='Heise', extension='xml', **kwargs):
    language = kwargs.get('language', 'en')
    normalization = kwargs.get('normalization', 'stemming')
    window = kwargs.get('window', 2)
    n_grams = kwargs.get('n_grams', 1)
    stoplist = kwargs.get('stoplist', None)
    batch_size = kwargs.get('batch_size', 100)
    num_documents = kwargs.get('num_documents', 0)
    co_occurrences = dict()
    word_counts = dict()
    num_words_total = 0

    if input_dir is None:
        print("No input directory set, loading documents from the database for the %s dataset." % dataset)
        db_handler = DatabaseHandler()

        if num_documents == 0:
            num_documents = db_handler.get_num_documents()

        while(num_documents > 0):
            documents, _ = db_handler.load_documents_from_db(KeyCluster, **kwargs)
            for key, doc in documents.items():
                logging.info('reading file ' + key)

                word_counts, co_occurrences, num_words_total = _compute_document_cooccurrence(output_file, key, doc['document'],
                                                                                              stoplist, n_grams, window,
                                                                                              word_counts, co_occurrences,
                                                                                              num_words_total)
            num_documents -= batch_size
            print("Done with batch.")
    else:
        for input_file in glob.glob(input_dir + "/*." + extension):
            logging.info('reading file ' + input_file)
            doc = pke.base.LoadFile()
            doc.load_document(input=input_file, language=language, normalization=normalization)

            word_counts, co_occurrences, num_words_total = _compute_document_cooccurrence(output_file, input_file, doc,
                                                                                          stoplist, n_grams, window,
                                                                                          word_counts, co_occurrences,
                                                                                          num_words_total)

    final_candidates = list(co_occurrences.keys())
    # Create global cooccurrence matrix from the dictionary
    cooccurrence_matrix = np.zeros((len(co_occurrences.keys()), len(co_occurrences.keys())))
    for word1, co_occurrence_words in co_occurrences.items():
        index1 = final_candidates.index(word1)
        for word2, co_occurrence_count in co_occurrence_words.items():
            index2 = final_candidates.index(word2)
            cooccurrence_matrix[index1][index2] = co_occurrence_count

    output = {
        'keys': final_candidates,
        'word_counts': word_counts,
        'num_words_total': num_words_total,
        'shape': cooccurrence_matrix.shape,
        'cooccurrence_matrix': cooccurrence_matrix.tolist(),
    }
    print(num_words_total)
    # print(cooccurrence_matrix.shape)
    # print(cooccurrence_matrix)
    # print(np.max(cooccurrence_matrix))
    with open(output_file, 'w') as f:
        json.dump(output, f)


def _compute_document_cooccurrence(output_file, doc_name, doc, stoplist, n_grams, window, word_counts, co_occurrences, num_words_total):
        logging.info('reading file ' + doc_name)

        # Get all candidates without stopwords
        if stoplist is None:
            stoplist = doc.stoplist
        doc.ngram_selection(n=n_grams)
        doc.candidate_filtering(stoplist=list(string.punctuation) +
                                             ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] + stoplist)

        # Get the global term frequency as well
        for word, values in doc.candidates.items():
            if word not in word_counts.keys():
                word_counts[word] = len(values.offsets)
            else:
                word_counts[word] += len(values.offsets)
        filtered_candidate_terms = list(doc.candidates.copy())

        # Gell all candidates but keep stopwords
        doc.candidates = defaultdict(Candidate)
        doc.ngram_selection(n=1)
        doc.candidate_filtering(stoplist=list(string.punctuation) +
                                             ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'],
                                    minimum_length=1, minimum_word_size=1, only_alphanum=False)
        cooccurrence_terms = list(doc.candidates.copy())

        # Calculate cooccurrence
        for sentence in list(doc.sentences):
            words = sentence.stems.copy()
            num_words_total += len(sentence.words)

            # Remove words/symbols that don't appear in the punctuation filtered tokens list
            for index in sorted([i for i, x in enumerate(words) if x not in cooccurrence_terms], reverse=True):
                words.pop(index)

            # Calculate the cooccurrence within a set window size
            for pos in range(len(words)):
                start = pos - window
                end = pos + window + 1

                # Skip stopwords
                if words[pos] not in filtered_candidate_terms:
                    continue

                if words[pos] not in co_occurrences.keys():
                    co_occurrences[words[pos]] = dict()

                if start < 0:
                    start = 0

                for word in words[start:pos] + words[pos + 1:end]:
                    # Skip stopwords
                    if word in filtered_candidate_terms:
                        if word in co_occurrences[words[pos]].keys():
                            co_occurrences[words[pos]][word] += 1
                        else:
                            co_occurrences[words[pos]][word] = 1

        return word_counts, co_occurrences, num_words_total


def load_global_cooccurrence_matrix(**kwargs):
    global_cooccurrence_matrix_path = kwargs.get('global_cooccurrence_matrix', None)
    cluster_feature_calculator = kwargs.get('cluster_feature_calculator', CooccurrenceClusterFeature)

    if global_cooccurrence_matrix_path is None or cluster_feature_calculator not in [CooccurrenceClusterFeature, PPMIClusterFeature]:
        print("No global cooccurrence matrix loaded.")

    else:
        with open(global_cooccurrence_matrix_path, 'r') as f:
            global_cooccurrence_matrix = json.load(f)
            global_cooccurrence_matrix['cooccurrence_matrix'] = np.array(
                global_cooccurrence_matrix['cooccurrence_matrix'])

        print("Finished loading global cooccurence matrix: %s" % global_cooccurrence_matrix_path)
        kwargs['global_cooccurrence_matrix'] = global_cooccurrence_matrix

    return kwargs


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


def _repel_labels(ax, x, y, labels, k=0.01):
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3",
                                    color='red'), )
    # expand limits
    all_pos = np.vstack(pos.values())
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos-x_span*0.15, 0)
    maxs = np.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])


def _create_simple_embedding_visualization(data, labels, selected_candidates, doc, filename):
    labels = list(labels)
    data = np.append(data, doc, axis=0)
    colors = ['c'] * (len(labels))
    for i in selected_candidates:
        colors[i] = 'g'

    # Transform the data into lesser dimensions
    n_components = 5
    pca = PCA(n_components=n_components).fit(data)
    pca_2d = pca.transform(data)
    x = pca_2d[:, 3]
    y = pca_2d[:, 4]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(x[:-1], y[:-1], marker='o', c=colors, s=90)
    ax.scatter(x[-1], y[-1], marker='*', c='r', s=135)
    # for i, txt in enumerate(labels):
    #     ax.annotate(txt, (x[i], y[i]))

    _repel_labels(ax, x, y, labels, k=0.21)

    ax.set_xlim(-2, 2.6)
    ax.set_ylim(-2, 2)
    plt.xticks(np.arange(-2, 3.0, 0.5))
    plt.yticks(np.arange(-2, 2, 0.5))

    i = len(labels)
    ax.annotate('Document', (x[i], y[i]), fontsize=14)

    markers = []
    markers.append(mlines.Line2D([], [], color='r', marker='*', linestyle='None',
                                 markersize=10, label='Document'))
    markers.append(mlines.Line2D([], [], color='c', marker='o', linestyle='None',
                                 markersize=10, label='Candidate Keyphrases'))
    markers.append(mlines.Line2D([], [], color='g', marker='o', linestyle='None',
                                 markersize=10, label='Selected Keyphrases'))
    plt.legend(handles=markers, loc='upper left', prop={'size': 14})

    plt.title('Simplified Visualization', fontdict={'fontsize': 20})
    plt.tight_layout()
    fig.savefig(os.path.basename(filename).split('.')[0] + ".pdf")
    plt.close()
