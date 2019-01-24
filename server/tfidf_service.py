class TfIdfService:
    def create_run_model(self, document, run):
        run_model = {}

        candidate_keywords = self.get_candidate_keyphrases_dict(run['candidate_keyphrases'])

        for sentence in document['sentences']:
            section = run_model.get(sentence['section'], [])
            sentence_model = self.create_sentence_model(sentence, candidate_keywords)
            section.append(sentence_model)
            run_model[sentence['section']] = section

        return run_model

    def create_sentence_model(self, sentence, candidate_keyphrases):
        sentence_model = []

        for index, word in enumerate(sentence['words']):
            if word.lower() in candidate_keyphrases:
                weights =  candidate_keyphrases[word.lower()]
                keyphrase_model = self.create_keyphrase_model(word, sentence, index, weights)
                sentence_model.append(keyphrase_model)
            else:
                word_model = self.create_word_model(word, sentence['POS'][index], [])
                sentence_model.append({'words': [word_model]})

        return sentence_model

    @staticmethod
    def get_matching_keyphrase(sentence_keyphrases, word_offsets):
        for keyphrase in sentence_keyphrases:
            keyphrase_start = keyphrase['char_offsets'][0][0]
            keyphrase_end = keyphrase['char_offsets'][-1][1]

            word_start, word_end = word_offsets

            if keyphrase_start <= word_start and word_end <= keyphrase_end:
                return keyphrase

        return None

    @staticmethod
    def get_candidate_keyphrases_dict(candidate_keyphrases):
        res = {}

        for candidate_keyphrase, weight in candidate_keyphrases:
            keywords = candidate_keyphrase.split()
            for keyword in keywords:
                weight_list = res.get(keyword, [])
                weight_list.append(str(weight))
                res[keyword] = weight_list

        return res

    @staticmethod
    def create_word_model(word, pos, weights):
        word_model = {
            'word': word,
            'properties': {
                'pos': pos,
            }
        }

        if weights:
            word_model['properties']['weight'] = ','.join(weights)

        return word_model

    def create_keyphrase_model(self, word, sentence, index, weights):
        keyphrase_model = {
            'properties': {
                'candidate_keyphrase': True,
                'candidate_keyphrase_selected': True
            },
        }

        words_model = [self.create_word_model(word, sentence['POS'][index], weights)]
        keyphrase_model['words'] = words_model

        return keyphrase_model
