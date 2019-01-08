def create_document_view_model(document):
    runs = []

    for run in document['runs']:
        run_model = {}

        candidate_keyphrases = get_candidate_keyphrases_dict(run['candidate_keyphrases'])

        for sentence in document['sentences']:
            section = run_model.get(sentence['section'], [])
            sentence_keyphrases = candidate_keyphrases[int(sentence['id'])]

            sentence_model = create_sentence_model(sentence, sentence_keyphrases)
            section.append(sentence_model)
            run_model[sentence['section']] = section

        runs.append(run_model)

    view_model = {
        'runs': runs
    }

    return view_model


def create_sentence_model(sentence, sentence_keyphrases):
    sentence_model = []
    matched_keyphrases = []

    for index, word in enumerate(sentence['words']):
        word_start = sentence['char_offsets'][index][0]
        word_end = sentence['char_offsets'][index][1]
        matching_keyphrase = get_matching_keyphrase(sentence_keyphrases, (word_start, word_end))

        if matching_keyphrase is not None:
            matching_keyphrase_end = matching_keyphrase['char_offsets'][-1][1]

            if matching_keyphrase_end not in matched_keyphrases:
                keyphrase_model = create_keyphrase_model(matching_keyphrase, sentence, index)
                sentence_model.append(keyphrase_model)
                matched_keyphrases.append(matching_keyphrase_end)

        else:
            word_model = create_word_model(word, sentence['POS'][index])
            sentence_model.append({'words': [word_model]})

    return sentence_model


def get_matching_keyphrase(sentence_keyphrases, word_offsets):
    for keyphrase in sentence_keyphrases:
        keyphrase_start = keyphrase['char_offsets'][0][0]
        keyphrase_end = keyphrase['char_offsets'][-1][1]

        word_start, word_end = word_offsets

        if keyphrase_start <= word_start and word_end <= keyphrase_end:
            return keyphrase

    return None


def get_candidate_keyphrases_dict(candidate_keyphrases):
    res = {}

    for _, candidate_keyphrase in candidate_keyphrases.items():
        sentence_keyphrases = res.get(candidate_keyphrase['sentence_id'], [])
        sentence_keyphrases.append(candidate_keyphrase)
        res[candidate_keyphrase['sentence_id']] = sentence_keyphrases

    return res


def create_word_model(word, pos):
    word_model = {
        'word': word,
        'pos': pos
    }

    return word_model


def create_keyphrase_model(matching_keyphrase, sentence, index):
    keyphrase_model = {
        'properties': {
            'candidate_keyphrase': True,
            'candidate_keyphrase_selected': matching_keyphrase['selected']
        },
    }

    words_model = []

    for i, word in enumerate(matching_keyphrase['words']):
        words_model.append(create_word_model(word, sentence['POS'][index+i]))

    keyphrase_model['words'] = words_model

    return keyphrase_model