from key_cluster_service import KeyClusterService

kc_service = KeyClusterService()
ti_service = TfIdfService()

def create_document_view_model(document):
    runs = []

    for run in document['runs']:
        run_model = {}

        if run['settings']['model'] == 'KeyCluster':
            run_model = kc_service.create_run_model(document, run)
            runs.append(run_model)

    view_model = {
        'runs': runs
    }

    return view_model