#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke
import numpy as np
import sklearn
import argparse
import configparser
from ast import literal_eval as make_tuple

from pke.unsupervised import TopicRank, MultipartiteRank, TfIdf

from common.CandidateSelector import key_cluster_candidate_selector, CandidateSelector
from common.ClusterFeatureCalculator import PPMIClusterFeature, WordEmbeddingsClusterFeature, CooccurrenceClusterFeature
from common.KeyphraseExtractor import KeyphraseExtractor
from common.helper import custom_normalize_POS_tags
from methods.KeyCluster import KeyCluster
from methods.EmbedRank import EmbedRank

pke.base.ISO_to_language['de'] = 'german'
pke.LoadFile.normalize_pos_tags = custom_normalize_POS_tags


def extract_keyphrases_from_raw_text(kwargs, input_document):
    extractor = KeyphraseExtractor()
    keyphrases, unstemmed_keyphrases = extractor.extract_keyphrases_from_raw_text(input_document=input_document, **kwargs)

    print("")
    print(keyphrases)
    # Additionally print the unstemmed keyphrases if stemming was used
    if kwargs.get('normalization', 'stemming') == 'stemming':
        print("\n")
        print(unstemmed_keyphrases)


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    # Create a dictionary from the config file
    kwargs = dict()
    for s in config.sections():
        for key, val in config.items(s):
            if len(val) > 0:
                kwargs[key] = val

    # This is needed for a few global statistic data.
    # We only provide the data for the German Heise dataset and the English Inspec dataset.
    if kwargs['language'] == 'de':
        kwargs['input_data'] = 'Heise'
    elif kwargs['language'] == 'en':
        kwargs['input_data'] = 'Inspec'

    if 'n_keyphrases' in kwargs.keys():
        kwargs['n_keyphrases'] = config['DEFAULT'].getint('n_keyphrases')

    # Set the model object and a few default settings that should NOT be changed by the user
    if kwargs['model'] == 'EmbedRank':
        kwargs['model'] = EmbedRank

        kwargs['document_similarity'] = config['DocSim'].getboolean('document_similarity')
        kwargs['document_similarity_new_candidate_constant'] = config['DocSim'].getfloat('document_similarity_new_candidate_constant')
        kwargs['document_similarity_weights'] = make_tuple(kwargs['document_similarity_weights'])

        kwargs['global_covariance'] = config['Mahalanobis'].getboolean('global_covariance')
        kwargs['global_covariance_weights'] = make_tuple(kwargs['global_covariance_weights'])

    elif kwargs['model'] == 'KeyCluster':
        kwargs['model'] = KeyCluster
        kwargs['candidate_selector'] = CandidateSelector(key_cluster_candidate_selector)

        # Set the feature type
        if kwargs['cluster_feature_calculator'] == 'PPMI':
            kwargs['cluster_feature_calculator'] = PPMIClusterFeature
        elif kwargs['cluster_feature_calculator'] == 'WordEmbeddings':
            kwargs['cluster_feature_calculator'] = WordEmbeddingsClusterFeature
        elif kwargs['cluster_feature_calculator'] in ['Cooccurrence', 'GlobalCooccurrence']:
            kwargs['cluster_feature_calculator'] = CooccurrenceClusterFeature

        # Set the global cooccurrence matrix if needed
        if config['Clustering']['cluster_feature_calculator'] in ['PPMI', 'GlobalCooccurrence']:
            if kwargs['input_data'] == 'Heise':
                kwargs['global_cooccurrence_matrix'] = 'data/global_cooccurrence/heise_train.cooccurrence'
            elif kwargs['input_data'] == 'Inspec':
                kwargs['global_cooccurrence_matrix'] = 'data/global_cooccurrence/inspec_train.cooccurrence'

        # Set the word embedding comp func if needed
        if kwargs['word_embedding_comp_func'] == 'cosine':
            kwargs['word_embedding_comp_func'] = sklearn.metrics.pairwise.cosine_similarity
        elif kwargs['word_embedding_comp_func'] == 'dot':
            kwargs['word_embedding_comp_func'] = np.dot
        elif kwargs['word_embedding_comp_func'] == 'manhattan':
            kwargs['word_embedding_comp_func'] = sklearn.metrics.pairwise.manhattan_distances
        elif kwargs['word_embedding_comp_func'] == 'euclidean':
            kwargs['word_embedding_comp_func'] = sklearn.metrics.pairwise.euclidean_distances

        # Handle numbers
        kwargs['min_word_count'] = config.getint('Clustering', 'min_word_count')
        kwargs['factor'] = config.getfloat('Clustering', 'factor')
        kwargs['global_cooccurrence_constant'] = config.getfloat('Clustering', 'global_cooccurrence_constant')

    elif kwargs['model'] == 'TopicRank':
        kwargs['model'] = TopicRank

    elif kwargs['model'] == 'MultipartiteRank':
        kwargs['model'] = MultipartiteRank

    elif kwargs['model'] == 'TfIdf':
        kwargs['model'] = TfIdf
        if kwargs['language'] == 'de':
            kwargs['frequency_file'] = 'data/document_frequency/heise_train_df_counts.tsv.gz'
        elif kwargs['language'] == 'en':
            kwargs['frequency_file'] = 'data/document_frequency/semEval_train_df_counts.tsv.gz'

    kwargs['print_document_scores'] = config['MISC'].getboolean('print_document_scores')
    kwargs['write_to_db'] = config['MISC'].getboolean('write_to_db')

    print(kwargs)
    return kwargs


def main():
    # Overwrite a few functions and variables so that the german language can be supported
    pke.LoadFile.normalize_POS_tags = custom_normalize_POS_tags
    pke.base.ISO_to_language['de'] = 'german'

    parser = argparse.ArgumentParser(description="Argument for the keyphrase extraction demo.")
    parser.add_argument('conf', type=str, default="data/demo_confs/de_embedrank.ini", nargs='?',
                        help="Path to the .ini file that holds the configuration for this extraction run.")
    parser.add_argument('input', type=str, default="demo_input.txt", nargs='?',
                        help="Path to the .txt file that contains the plain input text.")
    args = parser.parse_args()

    kwargs = load_config(args.conf)
    extract_keyphrases_from_raw_text(kwargs, args.input)


if __name__ == '__main__':
    main()
