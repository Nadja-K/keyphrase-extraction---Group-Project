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
    'X': 'x',
    '.': 's'
}


class KeyphraseSelector:
    def select_candidate_keyphrases(self, sentences, regex='a*n+'):
        candidate_keyphrases = dict()
        for sentence_id, sentence in enumerate(list(sentences)):
            pos_tags = ""
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
                        'stemmed': stemmed_keyphrase_list,
                        'unstemmed': unstemmed_keyphrase_list,
                        'pos': pos_keyphrase_list,
                        'char_offsets': offset_keyphrase_list,
                        'exemplar_terms_count': 0,
                        'weight': 0
                    }
        return candidate_keyphrases

    def filter_candidate_keyphrases(self, candidate_keyphrases, candidate_terms, cluster_exemplar_terms):
        for keyphrase in list(candidate_keyphrases.keys()):
            for key, val in cluster_exemplar_terms.items():
                if candidate_terms[val['centroid_index']] in keyphrase:
                    candidate_keyphrases[keyphrase]['exemplar_terms_count'] += 1
                    candidate_keyphrases[keyphrase]['weight'] = 1

            if candidate_keyphrases[keyphrase]['exemplar_terms_count'] == 0:
                del candidate_keyphrases[keyphrase]

        return candidate_keyphrases
