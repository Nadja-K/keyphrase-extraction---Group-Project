from key_cluster_service import KeyClusterService
from tfidf_service import TfIdfService
from extraction_service import ExtractionService

kc_service = KeyClusterService()
ti_service = TfIdfService()
ex_service = ExtractionService()


def create_document_view_model(document):
    runs = []
    for run in document['runs']:
        run_model = {}
        if run['settings']['model'] == 'KeyCluster':
            run_model = kc_service.create_run_model(document, run)
        elif run['settings']['model'] == 'TfIdf':
            run_model = ti_service.create_run_model(document, run)
        # elif run['settings']['model'] == 'EmbedRank':
        else:
            # FIXME: test if other models work with this format
            run_model = ex_service.create_run_model(document, run)

        run_model['settings'] = run['settings']
        runs.append(run_model)

    view_model = {
        'runs': runs
    }

    return view_model