# SETUP

## Requirements
* python 3.6.7
* run pip install -r requirements.txt

## External dependencies
* pke https://github.com/boudinfl/pke: run pip install git+https://github.com/boudinfl/pke.git
* sent2vec https://github.com/epfml/sent2vec :run pip install git+https://github.com/epfml/sent2vec.git

## Directory structure
FIXME

## Other file dependencies
* English word embedding model: FIXME (+ how to create a scipy version of that)
* German word embedding model: FIXME (+ how to create a scipy version of that)

* English sent2vec model: FIXME
* German sent2vec model: FIXME

* English frequent word list: FIXME
* German frequent word list: FIXME

* Global statistic matrices: FIXME

* Download English spacy models: python -m spacy download en
* Download German spacy models: python -m spacy download de

## Extracting keyphrases from a raw text
* German example: FIXME
** python src/extract_keyphrases_demo.py [path to .ini conf file] [path to .txt text input file]
** We provide a sample input file: demo_input.txt

* English example: FIXME
