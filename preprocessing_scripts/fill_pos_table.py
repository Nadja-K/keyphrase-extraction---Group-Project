import argparse
import xml.etree.ElementTree as etree
import glob
import multiprocessing
import os
import rethinkdb as r
from tqdm import tqdm


def parse_xml(document_path):
    parser = etree.XMLParser()
    sentences = []
    tree = etree.parse(document_path, parser)
    for sentence in tree.iterfind('./document/sentences/sentence'):
        # get the character offsets
        starts = [int(u.text) for u in
                  sentence.iterfind("tokens/token/CharacterOffsetBegin")]
        ends = [int(u.text) for u in
                sentence.iterfind("tokens/token/CharacterOffsetEnd")]
        sentences.append({
            "words": [u.text for u in
                      sentence.iterfind("tokens/token/word")],
            "lemmas": [u.text for u in
                       sentence.iterfind("tokens/token/lemma")],
            "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
            "char_offsets": [(starts[k], ends[k]) for k in
                             range(len(starts))]
        })
        sentences[-1].update(sentence.attrib)

    filename = os.path.basenme(document_path)
    id = os.path.splitext(filename)[0]

    parsed_document = {'id': id, 'sentences': sentences}

    return parsed_document


def insert_document(document_path, host, port):
    parsed_document = parse_xml(document_path)

    with r.connect(host, port, db='keyphrase_extraction') as conn:
        r.table('pos_tags').insert(parsed_document).run(conn)


def fill_pos_tags_table(input_dir, host, port, jobs):
    #mp_ctx = multiprocessing.get_context('forkserver')
    #pool = mp_ctx.Pool(processes=jobs)
    pool = multiprocessing.Pool(processes=jobs)

    input_dir_glob = os.path.join(input_dir, '*.xml')
    document_paths = glob.glob(input_dir_glob)

    with tqdm(total=len(document_paths)) as pbar:
        for document_path in document_paths:
            pool.apply_async(insert_document, (document_path, host, port), callback=lambda x: pbar.update())

        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fill document table')
    parser.add_argument('input_dir')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=28015)
    parser.add_argument('--jobs', type=int, default=1)

    args = parser.parse_args()
    fill_pos_tags_table(args.input_dir, args.host, args.port, args.jobs)
