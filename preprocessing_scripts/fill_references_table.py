import argparse
import glob
import json
import os
import rethinkdb as r
from tqdm import tqdm


def batch(iterable, size=1):
    length = len(iterable)
    for i in range(0, length, size):
        yield iterable[i:min(i + size, length)]


def insert_references(references, host, port, batchsize):
    conn = r.connect(host, port)
    table = r.db('keyphrase_extraction').table('references')

    with tqdm(total=len(references)) as pbar:
        for entry in batch(references, batchsize):
            table.insert(entry).run(conn)
            pbar.update(batchsize)

    conn.close()


def fill_references_table(input_dir, host, port, batchsize):
    input_dir_glob = os.path.join(input_dir, '*.json')

    """
    [id:{
    'exact_filtered_stemmed': [a,b,c],
    'exact_filtered_unstemmed': [d,e,f],
    'stemmed_filtered_stemmed': [g,h,i]
    }]
    """

    references = {}

    for document_path in glob.glob(input_dir_glob):
        filename = os.path.basename(document_path)
        document_name = os.path.splitext(filename)[0]
        references_config_name = document_name.replace('heise_', '')

        with open(document_path) as f:
            document = json.load(f)

        for id, data in document.items():
            id_obj = references.get(id, {})

            flat_data = [item for sublist in data for item in sublist]
            id_obj[references_config_name] = flat_data

            references[id] = id_obj

    references_list = []
    for id, data in references.items():
        data['id'] = id
        references_list.append(data)

    insert_references(references_list, host, port, batchsize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fill document table')
    parser.add_argument('input_dir')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=28015)
    parser.add_argument('--batchsize', type=int, default=50)

    args = parser.parse_args()
    fill_references_table(args.input_dir, args.host, args.port, args.batchsize)
