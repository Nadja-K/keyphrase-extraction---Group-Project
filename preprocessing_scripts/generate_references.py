import argparse
import glob
import json
import os
import multiprocessing
from functools import partial
from nltk.stem.snowball import SnowballStemmer as Stemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

_special_chars_map = {i: '\\' + chr(i) for i in b'()[]{}?*+-|^$\\.&~#\t\n\r\v\f'}


def get_stemmer():
    return Stemmer('german')


def rolling_window(iterable, size):
    for i in range(len(iterable)-size+1):
        yield iterable[i:i+size]


def is_exact_match(substring, string):
    substring_tokens = word_tokenize(substring, 'german')
    string_tokens = word_tokenize(string, 'german')

    for string_window in rolling_window(string_tokens, len(substring_tokens)):
        if substring_tokens == string_window:
            return True

    return False


def filter_keyphrases(keyphrases, text, filter):
    filtered_keyphrases = []

    if filter == 'exact':
        text_lower =  text.lower()
        for keyphrase in keyphrases:
            if is_exact_match(keyphrase.lower(), text_lower):
                filtered_keyphrases.append(keyphrase)

    elif filter == 'stemmed':
        stemmer = get_stemmer()

        stemmed_text_words = [stemmer.stem(word) for word in word_tokenize(text, 'german')]
        stemmed_text = ' '.join(stemmed_text_words)

        for keyphrase in keyphrases:
            stemmed_keyphrase_words = [stemmer.stem(word) for word in keyphrase.split()]
            stemmed_keyphrase = ' '.join(stemmed_keyphrase_words)

            if is_exact_match(stemmed_keyphrase, stemmed_text):
                filtered_keyphrases.append(keyphrase)

    else:
        filtered_keyphrases = keyphrases

    return filtered_keyphrases


def process_keyphrases(keyphrases, text, stem, filter):
    stemmer = get_stemmer()

    processed_keyphrases = []

    filtered_keyphrases = filter_keyphrases(keyphrases, text, filter)

    for keyphrase in filtered_keyphrases:
        words = keyphrase.split()

        if stem:
            words = [stemmer.stem(word) for word in words]

        words = [word.lower() for word in words]
        processed_keyphrase = ' '.join(words)
        processed_keyphrases.append(processed_keyphrase)

    return processed_keyphrases

def generate_entry_references(entry, stem, filter):
    id = entry['id']

    if 'keyword' in entry:
        keywords = entry['keyword']
    else:
        keywords = []

    if 'related' in entry:
        related = entry['related']
    else:
        related = []

    text = ' '.join([entry['headline'], entry['lead'], entry['text']])

    keyphrases = set(keywords) | set(related)

    if not keyphrases:
        return id, None

    processed_keyphrases = process_keyphrases(keyphrases, text, stem, filter)
    if not processed_keyphrases:
        return id, None

    return id, [[k] for k in processed_keyphrases]


def generate_document_references(document, stem, filter):
    references = {}

    for entry in document:
        id = entry['id']

        if 'keyword' in entry:
            keywords = entry['keyword']
        else:
            keywords = []

        if 'related' in entry:
            related = entry['related']
        else:
            related = []

        text = ' '.join([entry['headline'], entry['lead'], entry['text']])

        keyphrases = set(keywords) | set(related)

        if not keyphrases:
            continue

        processed_keyphrases = process_keyphrases(keyphrases, text, stem, filter)
        if processed_keyphrases:
            references[id] = [[k] for k in processed_keyphrases]

    return references

def generate_references2(input_dir, output_path, stem, filter, jobs):
    input_dir_glob = os.path.join(input_dir, '*.json')

    references = {}

    for document_path in tqdm(glob.glob(input_dir_glob)):
        with open(document_path) as f:
            document = json.load(f)

        document_references = generate_document_references(document, stem, filter)
        references.update(document_references)

    with open(output_path, 'w') as file:
        json.dump(references, file)

def generate_references(input_dir, output_path, stem, filter, jobs):
    input_dir_glob = os.path.join(input_dir, '*.json')

    references = {}

    mp_ctx = multiprocessing.get_context('forkserver')
    pool = mp_ctx.Pool(processes=jobs)

    for document_path in tqdm(glob.glob(input_dir_glob)):
        with open(document_path) as f:
            document = json.load(f)

        document_references = pool.map(partial(generate_entry_references, stem=stem, filter=filter), tqdm(document), chunksize=jobs * 10)

        for entry_id, entry_references in document_references:
            if entry_references:
                references[entry_id] = entry_references

    with open(output_path, 'w') as file:
        json.dump(references, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate references')
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument('--stem',  action='store_true')
    parser.add_argument('--filter', default='none')
    parser.add_argument('--jobs', type=int, default=1)

    args = parser.parse_args()
    print(args.stem, args.filter)
    generate_references(args.input_dir, args.output_path, args.stem, args.filter, args.jobs)