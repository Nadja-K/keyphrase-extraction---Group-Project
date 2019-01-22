class KeyClusterService:
    def create_run_model(self, document, run):
        run_model = {}

        candidate_keyphrases = self.get_candidate_keyphrases_dict(run['candidate_keyphrases'])
        cluster_members = self.get_cluster_members_dict(run['cluster_members'])
        run_model['num_clusters'] = max(cluster_members.values(), key=lambda v: v['cluster'])['cluster']

        for sentence in document['sentences']:
            section = run_model.get(sentence['section'], [])
            sentence_keyphrases = candidate_keyphrases.get(int(sentence['id']), [])

            sentence_model = self.create_sentence_model(sentence, sentence_keyphrases, cluster_members)
            section.append(sentence_model)
            run_model[sentence['section']] = section

        return run_model

    def create_sentence_model(self, sentence, sentence_keyphrases, cluster_members):
        sentence_model = []
        matched_keyphrases = []

        for index, word in enumerate(sentence['words']):
            word_start = sentence['char_offsets'][index][0]
            word_end = sentence['char_offsets'][index][1]
            matching_keyphrase = self.get_matching_keyphrase(sentence_keyphrases, (word_start, word_end))

            if matching_keyphrase is not None:
                matching_keyphrase_end = matching_keyphrase['char_offsets'][-1][1]

                if matching_keyphrase_end not in matched_keyphrases:
                    keyphrase_model = self.create_keyphrase_model(matching_keyphrase, sentence, index, cluster_members)
                    sentence_model.append(keyphrase_model)
                    matched_keyphrases.append(matching_keyphrase_end)

            else:
                cluster_member = cluster_members.get(word, None)
                word_model = self.create_word_model(word, sentence['POS'][index], cluster_member)
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

        for _, candidate_keyphrase in candidate_keyphrases.items():
            sentence_keyphrases = res.get(candidate_keyphrase['sentence_id'], [])
            sentence_keyphrases.append(candidate_keyphrase)
            res[candidate_keyphrase['sentence_id']] = sentence_keyphrases

        return res

    @staticmethod
    def get_cluster_members_dict(cluster_members):
        res = {}

        for _, cluster_member in cluster_members.items():
            surface_forms = set(x[0] for x in cluster_member['surface_forms'])
            for surface_form in surface_forms:
                res.update({
                    surface_form:
                    {
                        'cluster': cluster_member['cluster'],
                        'exemplar_term': cluster_member['exemplar_term']
                    }
                })

        return res

    @staticmethod
    def create_word_model(word, pos, cluster_member):
        word_model = {
            'word': word,
            'properties': {
                'pos': pos
            }
        }

        if cluster_member:
            word_model['properties']['cluster'] = cluster_member['cluster']
            word_model['properties']['exemplar_term'] = cluster_member['exemplar_term']

        return word_model

    def create_keyphrase_model(self, matching_keyphrase, sentence, index, cluster_members):
        keyphrase_model = {
            'properties': {
                'candidate_keyphrase': True,
                'candidate_keyphrase_selected': matching_keyphrase['selected']
            },
        }

        words_model = []

        for i, word in enumerate(matching_keyphrase['words']):
            cluster_member = cluster_members.get(word, None)
            words_model.append(self.create_word_model(word, sentence['POS'][index + i], cluster_member))

        keyphrase_model['words'] = words_model

        return keyphrase_model
