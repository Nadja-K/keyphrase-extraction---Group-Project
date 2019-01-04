import rethinkdb as r
from pke.data_structures import Document
import pke
from nltk.corpus import stopwords


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
            cursor = r.table('references').order_by(index=r.desc('id')).has_fields(reference_table).pluck(reference_table, 'id').eq_join(
                'id', r.table(table), ordered=True).zip().slice(self._current_index, self._current_index + batch_size).run(conn)
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
