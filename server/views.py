from aiohttp import web
from repository import get_documents, get_document
from service import create_document_view_model

async def index(request):
    return web.Response(text='Hello Aiohttp!')


async def documents(request):
    pagesize = int(request.app['config']['api']['page_size'])

    dataset = request.match_info['dataset']
    page = int(request.query.get('page', 0))

    start, end = get_bounds(page, pagesize)
    documents = await get_documents(request, dataset, start, end)

    documents = [
        {
            'id': document['id'],
            'headline': document['headline'],
            'text': document['lead'] if document['lead'] else document['text'],
        }
        for document in documents
    ]

    return web.json_response(documents)


async def document(request):
    dataset = request.match_info['dataset']
    document_id = int(request.match_info['id'])
    document = await get_document(request, dataset, document_id)
    view_model = create_document_view_model(document)
    return web.json_response(view_model)

def get_bounds(page, pagesize):
    start = page*pagesize
    end = start + pagesize

    return start, end