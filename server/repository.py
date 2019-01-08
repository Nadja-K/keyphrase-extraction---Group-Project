import rethinkdb as r


async def get_documents(request, dataset, start, end):
    documents = []

    conn = await request.app['db']
    cursor = await r.table('documents').order_by(index='id').slice(start, end).pluck('id', 'headline', 'lead', 'text').run(conn)
    while await cursor.fetch_next():
        item = await cursor.next()
        documents.append(item)

    return documents


async def get_document(request, dataset, document_id):
    conn = await request.app['db']
    # Todo handle invalid id
    document = await (
        r.table('pos_tags').get(str(document_id)).merge(
            r.table('document_runs').get(document_id)
        ).default(None).run(conn)
    )

    return document

async def get_document2(request, dataset, document_id):
    conn = await request.app['db']
    document = await r.table('documents').get(document_id).run(conn)
    # Todo handle invalid id
    return document