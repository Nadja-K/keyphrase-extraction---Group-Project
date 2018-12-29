import re

_MAPPING = {
    'NOUN': 'n',
    'VERB': 'v',
    'PRON': 'p',
    'ADJ': 'a',
    'ADV': 'ä',
    'ADP': 'A',
    'CONJ': 'c',
    'DET': 'd',
    'NUM': 'N',
    'PRT': 'P',
    'PROPN': 'q',
    'X': 'x',
    '.': 's',
    'PUNCT': 's',
    'SPACE': ''
}


class KeyphraseSelector:
    def select_candidate_keyphrases(self, sentences, regex='a*n+'):
        candidate_keyphrases = dict()
        for sentence_id, sentence in enumerate(list(sentences)):
            sentence_id += 1
            pos_tags = ""
            # Create a simpler string representing the PoS Tags for the regex
            for pos in sentence.pos:
                pos_tags = pos_tags + _MAPPING[pos]

            # Use the regex from the original paper (JJ)*(NN|NNS|NNP)+
            for match in re.finditer(regex, pos_tags):
                start = match.start()
                end = match.end()
                word_indices = range(start, end)

                keyphrase = ""
                stemmed_keyphrase_list = []
                unstemmed_keyphrase_list = []
                pos_keyphrase_list = []
                offset_keyphrase_list = []
                for word_index in word_indices:
                    keyphrase += sentence.stems[word_index] + " "

                    stemmed_keyphrase_list.append(sentence.stems[word_index])
                    unstemmed_keyphrase_list.append(sentence.words[word_index])
                    pos_keyphrase_list.append(sentence.pos[word_index])
                    offset_keyphrase_list.append(sentence.meta['char_offsets'][word_index])

                keyphrase = keyphrase[:-1]
                if keyphrase not in candidate_keyphrases:
                    candidate_keyphrases[keyphrase] = {
                        'sentence_id': sentence_id,
                        'stems': stemmed_keyphrase_list,
                        'words': unstemmed_keyphrase_list,
                        'pos': pos_keyphrase_list,
                        'char_offsets': offset_keyphrase_list,
                        'exemplar_terms_count': 0,
                        'weight': 0
                    }
        return candidate_keyphrases

    def filter_candidate_keyphrases(self, candidate_keyphrases, candidate_terms, cluster_exemplar_terms):
        candidate_terms = list(candidate_terms)
        unfiltered_candidate_keyphrases = candidate_keyphrases.copy()

        for keyphrase in list(candidate_keyphrases.keys()):
            for key, val in cluster_exemplar_terms.items():
                candidate_term = candidate_terms[val['centroid_index']]

                if candidate_term in keyphrase.split(' '):
                    candidate_keyphrases[keyphrase]['exemplar_terms_count'] += 1
                    candidate_keyphrases[keyphrase]['weight'] = 1
                    unfiltered_candidate_keyphrases[keyphrase].update(candidate_keyphrases[keyphrase])

            if candidate_keyphrases[keyphrase]['exemplar_terms_count'] == 0:
                del candidate_keyphrases[keyphrase]

        return candidate_keyphrases, unfiltered_candidate_keyphrases

    def frequent_word_filtering(self, frequent_word_list, candidate_keyphrases):
        """
        Filter out single word keyphrases that are part of the frequent_word_list

        :param frequent_word_list:
        :return:
        """
        if len(frequent_word_list) == 0:
            return candidate_keyphrases
        else:
            for keyphrase in list(candidate_keyphrases.keys()):
                if keyphrase in frequent_word_list:
                    del candidate_keyphrases[keyphrase]
                    # print(keyphrase + " deleted")
            return candidate_keyphrases
