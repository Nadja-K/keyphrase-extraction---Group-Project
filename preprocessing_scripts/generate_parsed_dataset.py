from nltk.parse import CoreNLPParser
import xml.etree.ElementTree as ET
import argparse
import glob
import os
import json
import multiprocessing
from functools import partial
from tqdm import tqdm
import logging

tqdm.monitor_interval = 0

def parse(nlp_parser, data):
    properties = {
        'annotators': 'tokenize,ssplit,pos,lemma',
        'ssplit.newlineIsSentenceBreak': 'always',
        'outputFormat': 'xml'}

    response = nlp_parser.session.post(
        nlp_parser.url,
        params={'properties': json.dumps(properties)},
        data=data.encode(nlp_parser.encoding),
        timeout=60
    )

    response.raise_for_status()
    return response.content


def get_sentences(elem):
    return list(elem.findall('.//sentence'))


def set_section(sentences, section):
    for sentence in sentences:
        sentence.set('section', section)


def offset_sentence_ids(sentences, offset):
    for sentence in sentences:
        id = int(sentence.get('id'))
        sentence.set('id', str(id + offset))


def insert_sentences(sentences, elem):
    for index, sentence in enumerate(sentences):
        elem.insert(index, sentence)


def remove_abstract(text, abstract):
    index = text.find(abstract)

    if index != -1:
        text = text[index+len(abstract):].lstrip()

    return text


def parse_entry(entry, output_dir, url, override):

    output_path = os.path.join(output_dir, str(entry['id']) + '.xml')

    if not override and os.path.exists(output_path):
        return

    title = entry['headline']
    abstract = entry['lead']
    text = entry['text'].replace('%00', '00').replace('%01', '01')

    text = remove_abstract(text, abstract)

    nlp_parser = CoreNLPParser(url=url)

    try:
        parsed_title = ET.fromstring(parse(nlp_parser, title))
        parsed_abstract = ET.fromstring(parse(nlp_parser, abstract))
        parsed_text = ET.fromstring(parse(nlp_parser, text))

        title_sentences = get_sentences(parsed_title)
        abstract_sentences = get_sentences(parsed_abstract)
        text_sentences = get_sentences(parsed_text)

        set_section(title_sentences, 'title')
        set_section(abstract_sentences, 'abstract')
        set_section(text_sentences, 'text')

        offset_sentence_ids(abstract_sentences, len(title_sentences))
        offset_sentence_ids(text_sentences, len(title_sentences) + len(abstract_sentences))

        insert_sentences(title_sentences + abstract_sentences, parsed_text.find('.//sentences'))

        ET.ElementTree(parsed_text).write(output_path, encoding='UTF-8', xml_declaration=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        logging.exception('Error in entry {0}'.format(entry['id']))

def parse_documents(input_dir, output_dir, url, jobs, override):
    input_dir_glob = os.path.join(input_dir, '*.json')

    #mp_ctx = multiprocessing.get_context('forkserver')
    pool = multiprocessing.Pool(processes=jobs)

    for document_path in tqdm(glob.glob(input_dir_glob)):
        tqdm.write('Begin processing of document {0}'.format(document_path))

        try:
            with open(document_path) as f:
                document = json.load(f)

            pool.map(partial(parse_entry, output_dir=output_dir, url=url, override=override), tqdm(document), chunksize=jobs*3)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            #logging.exception('Error in document {0}'.format(document_path))
            tqdm.write('Error in document {0}'.format(document_path))

    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate parsed dataset')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--url', default='http://localhost:9000')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--override', type=bool, default=False)

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    parse_documents(args.input_dir, args.output_dir, args.url, args.jobs, args.override)