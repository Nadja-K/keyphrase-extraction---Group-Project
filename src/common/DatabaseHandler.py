import rethinkdb as r
from pke.data_structures import Document
import pke
from nltk.corpus import stopwords

from common.CandidateSelector import CandidateSelector
from common.KeyphraseSelector import KeyphraseSelector


class DatabaseHandler:
    def __init__(self, host='localhost', port=28015):
        self._host = host
        self._port = port
        self._current_index = 0

    def reset_current_index(self):
        self._current_index = 0

    def get_num_documents(self):
        with r.connect(self._host, self._port, db='keyphrase_extraction') as conn:
            num_documents = r.table('documents').count().run(conn)

            return num_documents

    def get_num_documents_with_keyphrases(self, **kwargs):
        reference_table = kwargs.get('reference_table', 'exact_filtered_stemmed')
        with r.connect(self._host, self._port, db='keyphrase_extraction') as conn:
            num_documents = r.table('references').has_fields(reference_table).count().run(conn)

            return num_documents

    def write_data_to_db(self, filename, doc_eval_data, data_cluster_members=[], data_candidate_keyphrases=[], **settings):
        reformatted_settings = settings.copy()

        # Remove unneeded data
        if 'frequent_word_list' in reformatted_settings.keys():
            del(reformatted_settings['frequent_word_list'])
        if 'word_embedding_model' in reformatted_settings.keys():
            del(reformatted_settings['word_embedding_model'])
        if 'global_cooccurrence_matrix' in reformatted_settings.keys():
            del(reformatted_settings['global_cooccurrence_matrix'])

        # Adjust the settings so that the information can be written to the database
        comp_funcs = []
        for comp_func in reformatted_settings['evaluator_compare_func']:
            comp_funcs.append(comp_func.__name__)
        reformatted_settings['evaluator_compare_func'] = comp_funcs

        for key, val in reformatted_settings.items():
            if callable(val) is True:
                reformatted_settings[key] = val.__name__
            elif isinstance(val, CandidateSelector) or isinstance(val, KeyphraseSelector):
                reformatted_settings[key] = val.__class__.__name__

        run = {
            'eval_scores': doc_eval_data,
            'settings': reformatted_settings,
            'cluster_members': data_cluster_members,
            'candidate_keyphrases': data_candidate_keyphrases
        }
        self.write_document_run_to_db(filename, run)

    def write_document_run_to_db(self, id, run):
        id = int(id)
        with r.connect(self._host, self._port, db='keyphrase_extraction') as conn:
            cursor = r.table('document_runs').get_all(int(id)).run(conn)
            cursor = list(cursor)
            if len(cursor) == 0:
                runs = [run]
                data = {'id': id, 'runs': runs}
                cursor = r.table('document_runs').insert(data).run(conn)
                print("ID not found in table, creating new entry.")
            else:
                runs = cursor[0]['runs']
                runs.append(run)
                data = {'id': id, 'runs': runs}
                cursor = r.table('document_runs').update(data).run(conn)
                print("ID found in table, updating entry with new runs.")


    def load_documents_from_db(self, model, **kwargs):
        """
        Loads a set number of pre-parsed documents from the database and returns them as KeyCluster instances.

        :param model:
        :param batch_size:
        :param language:
        :param table:
        :return:
        """
        batch_size = kwargs.get('batch_size', 100)
        table = kwargs.get('table', 'pos_tags')
        reference_table = kwargs.get('reference_table', 'exact_filtered_stemmed')

        extracted_documents = dict()
        extracted_documents_references = dict()
        with r.connect(self._host, self._port, db='keyphrase_extraction') as conn:
            cursor = r.table('references').order_by(index=r.desc('id')).has_fields(reference_table).pluck(
                reference_table, 'id').slice(self._current_index, self._current_index + batch_size).eq_join(
                'id', r.table(table), ordered=True).zip().run(conn)
            # cursor = [r.table('pos_tags').get("2730613").merge(r.table('references').get("2730613")).run(conn)]
            for document in cursor:
                doc = Document.from_sentences(document['sentences'], **kwargs)
                doc.is_corenlp_file = True
                extractor = model()
                extractor.input_file = doc.input_file

                # set the language of the document
                extractor.language = kwargs.get('language', 'en')

                # set the sentences
                extractor.sentences = doc.sentences

                # initialize the stoplist
                extractor.stoplist = stopwords.words(pke.base.ISO_to_language[extractor.language])

                # word normalization
                extractor.normalization = kwargs.get('normalization', 'stemming')
                if extractor.normalization == 'stemming':
                    extractor.apply_stemming()
                elif extractor.normalization is None:
                    for i, sentence in enumerate(extractor.sentences):
                        extractor.sentences[i].stems = sentence.words

                # lowercase the normalized words
                for i, sentence in enumerate(extractor.sentences):
                    extractor.sentences[i].stems = [w.lower() for w in sentence.stems]

                # POS normalization
                if getattr(doc, 'is_corenlp_file', False):
                    extractor.normalize_pos_tags()
                    extractor.unescape_punctuation_marks()

                extracted_documents[document['id']] = {
                    'document': extractor,
                    'id': document['id']
                }
                extracted_documents_references[document['id']] = document[reference_table]
                self._current_index += 1

        return extracted_documents, extracted_documents_references
