from views import index, documents, document


def setup_routes(app):
    app.router.add_get('/', index)
    app.router.add_get('/{dataset}/documents', documents)
    app.router.add_get('/{dataset}/documents/{id}', document)