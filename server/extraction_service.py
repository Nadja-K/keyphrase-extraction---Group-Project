class ExtractionService():
    def create_run_model(self, document, run):
        run_model = {}

        candidate_keyphrases = self.get_candidate_keyphrases_dict(run['candidate_keyphrases'])

        for sentence in document['sentences']:
            section = run_model.get(sentence['section'], [])
            # FIXME: why is the sentence id off by 1?
            sentence_keyphrases = candidate_keyphrases.get(int(sentence['id'])-1, [])

            sentence_model = self.create_sentence_model(sentence, sentence_keyphrases)
            section.append(sentence_model)
            run_model[sentence['section']] = section

        return run_model

    def create_sentence_model(self, sentence, sentence_keyphrases):
        sentence_model = []
        matched_keyphrases = []
        for index, word in enumerate(sentence['words']):
            word_start = sentence['char_offsets'][index][0]
            word_end = sentence['char_offsets'][index][1]
            matching_keyphrase = self.get_matching_keyphrase(sentence_keyphrases, (word_start, word_end))

            if matching_keyphrase is not None:
                matched_keyphrase_end = matching_keyphrase['offsets']

                if matched_keyphrase_end not in matched_keyphrases:
                    keyphrase_model = self.create_keyphrase_model(matching_keyphrase, sentence, index)
                    sentence_model.append(keyphrase_model)
                    matched_keyphrases.append(matched_keyphrase_end)
            else:
                word_model = self.create_word_model(word, sentence['POS'][index], [])
                sentence_model.append({'words': [word_model]})

        return sentence_model

    @staticmethod
    def get_matching_keyphrase(keyphrases, word_offsets):
        for keyphrase in keyphrases:
            keyphrase_start = keyphrase['offsets'][0]
            keyphrase_end = keyphrase['offsets'][1]

            word_start, word_end = word_offsets

            if keyphrase_start <= word_start and word_end <= keyphrase_end:
                return keyphrase

        return None

    @staticmethod
    def get_candidate_keyphrases_dict(candidate_keyphrases):
        res = {}

        for _, candidate_keyphrase in candidate_keyphrases.items():
            candidate_sentence_ids = candidate_keyphrase['sentence_id']

            for index, sentence_id in enumerate(candidate_sentence_ids):
                sentence_keyphrases = res.get(sentence_id, [])

                sentence_candidate_keyphrase = candidate_keyphrase.copy()
                sentence_candidate_keyphrase['offsets'] = sentence_candidate_keyphrase['offsets'][index]
                sentence_candidate_keyphrase['sentence_id'] = sentence_candidate_keyphrase['sentence_id'][index]
                sentence_candidate_keyphrase['pos'] = sentence_candidate_keyphrase['pos'][index]
                sentence_candidate_keyphrase['words'] = sentence_candidate_keyphrase['words'][index]

                sentence_keyphrases.append(sentence_candidate_keyphrase)
                res[sentence_id] = sentence_keyphrases

        return res

    @staticmethod
    def create_word_model(word, pos, weight):
        word_model = {
            'word': word,
            'properties': {
                'pos': pos
            }
        }

        if weight:
            word_model['properties']['weight'] = weight

        return word_model

    def create_keyphrase_model(self, matching_keyphrase, sentence, index):
        keyphrase_model = {
            'properties': {
                'candidate_keyphrase': True,
                'candidate_keyphrase_selected': matching_keyphrase['selected']
            },
        }

        words_model = []

        for i, word in enumerate(matching_keyphrase['words']):
            words_model.append(self.create_word_model(word, sentence['POS'][index + i], matching_keyphrase['weight']))

        keyphrase_model['words'] = words_model

        return keyphrase_model
