import argparse
import glob
import json
import multiprocessing
import os
import rethinkdb as r
from tqdm import tqdm

tqdm.monitor_interval = 0


def batch(iterable, size=1):
    length = len(iterable)
    for i in range(0, length, size):
        yield iterable[i:min(i + size, length)]


def insert_document(document, host, port, batchsize):
    conn = r.connect(host, port)
    table = r.db('keyphrase_extraction').table('documents')

    for entries in batch(document, batchsize):
        table.insert(entries).run(conn)


def fill_document_table(input_dir, host, port, batchsize, jobs):
    input_dir_glob = os.path.join(input_dir, '*.json')

    #mp_ctx = multiprocessing.get_context('forkserver')
    #pool = mp_ctx.Pool(processes=jobs)
    pool = multiprocessing.Pool(processes=jobs)

    document_paths = glob.glob(input_dir_glob)

    with tqdm(total=len(document_paths)) as pbar:
        for document_path in document_paths:
            with open(document_path) as f:
                document = json.load(f)
                pool.apply_async(insert_document, (document, host, port, batchsize), callback=lambda x: pbar.update())

        pool.close()
        pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fill document table')
    parser.add_argument('input_dir')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=28015)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--jobs', type=int, default=1)

    args = parser.parse_args()
    fill_document_table(args.input_dir, args.host, args.port, args.batchsize, args.jobs)
