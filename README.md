# Unsupervised Keyphrase Extraction
TLDR: We re-implemented and modified two state-of-the-art unsupervised keyphrase extraction algorithms: KeyCluster and EmbedRank.


## Project structure
```
project
│   demo_input.txt
│   README.md
│
└───data
│   └───demo_confs
│   │   │   de_embedrank.ini
│   │   │   de_keycluster.ini
│   │   │   en_embedrank.ini
│   │   │   en_keycluster.ini
│   │      
│   └───document_similarity
│   │   │   Heise_similarity_dataframe          [Needs to be downloaded separately]
│   │   │   Inspec_similarity_dataframe         [Needs to be downloaded separately]
│   │   
│   └───mahalanobis_covariance
│   │   │   Heise_inv_covariance.npy           
│   │   │   Heise_inv_covariance_centroid.npy  
│   │   │   Inspec_inv_covariance.npy          
│   │   │   Inspec_inv_covariance_centroid.npy 
│   │   
│   └───frequent_word_lists
│   │   │   de_50k.txt
│   │   │   en_50k.txt
│   │
│   └───global_cooccurrence
│       │   heise_train.cooccurrence            [Needs to be downloaded separately]
│       │   inspec_train.cooccurrence           [Needs to be downloaded separately]
│
└───frontend
│   │   ...
│
└───preprocessing_scripts
│   │   ...
│
└───server
│   │   ...
│
└───setup
│   │   requirements.txt
│
└───src
│   │   extract_keyphrase_demo.py
│   │   heise_eval_main.py
│   │   others_eval_main.py
│   │
│   └───common
│   │   |   ...
│   │
│   └───eval
│   │   |   ...
│   │
│   └───methods
│   │   |   ...
│   │
│   └───preprocessing
│       |   ...
│
└───tests
    │   ...
```

The code for the implementation and modification of the two keyphrase extraction methods can be found under ```src```.
Furthermore, we implemented a visualization for extracted keyphrases. The code for this visualization demo can be found under ```frontend``` and ```server```.


## Setup & Requirements
This code requires python 3.6.7, as well as a few external dependencies ([see the next Section](#external-dependencies)).
To install the remaining requirements, create a new python environment and run:
```
pip install -r setup/requirements.txt
```
To support German and English texts in scipy, run the following commands:
```
python -m spacy download en
python -m spacy download de
```

## External dependencies
To install pke run:
```
pip install git+https://github.com/boudinfl/pke.git
```
and to install sent2vec run:
```
pip install git+https://github.com/epfml/sent2vec.git
```


## Other file dependencies
For the modifications, further (large) file dependencies are required, that can't be provided in this git repo. 
Instead follow these instructions:

### Word Embedding Models
Download an English word embedding model from [here](https://nlp.stanford.edu/projects/glove/).
We used the [Wikipedia 2014 + Gigaword5](http://nlp.stanford.edu/data/glove.6B.zip) word embedding model with 50 dimensions.
For the German word embedding model, we used the provided on <https://devmount.github.io/GermanWordEmbeddings/>.
The direct download link can be found [here](http://cloud.devmount.de/d2bc5672c523b086).

After downloading both models, follow these steps to create a scipy version that can be imported into the project.
For the German word embedding model run:
```
import gensim

trained_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/downloaded/model/german.model', binary=True)
trained_moodel.save_word2vec_format('path/to/reformatted/output/model/german.model.txt', binary=False)
```
Convert the saved model to a spacy compatible version by running:
```
python -m spacy init-model de output_directory --vectors-loc path/to/reformatted/output/model/german.model.txt
```

For the English word embedding model run the following code from within the ```src``` folder of the project:
```
from common.helper import reformat_glove_model
reformat_glove_model('path/to/downloaded/model/glove6B.50d.txt')
```
```
python -m spacy init-model en output_directory --vectors-loc path/to/reformatted/output/model/glove6B.50d.recoded.txt
```


### Sent2Vec Models
Download the following English sent2vec model from the [official sent2vec git](https://github.com/epfml/sent2vec): [sent2vec_wiki_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSaER5YkJUdWdPWU0)

An unofficial German (and French) sent2vec model can be found under: [sent2vec_pagano](https://drive.google.com/file/d/199WZvUYTDaOl-xAwhLowVNFFdv_2eiXF/view?usp=sharing)


### Frequent Word Lists
Various frequent word lists of different languages can be found under [this link](https://github.com/hermitdave/FrequencyWords/).
We also provide the two lists, that were used in this project, in the ```data``` directory.


### Global statistic matrices
We calculated various global statistics for our modifications, depending on the dataset used. Because of their size, we can not provide the original files in this git. Instead please download them from cuda2.
All files can be found in ```/video2/keyphrase_extraction/keyphrase_extraction_nadja/data/*```
The files in question are the ones marked in the [Project Structure](#project-structure) Section.


## Extracting keyphrases from a raw text
We provide a small script to extract keyphrases from a raw ```German``` or ```English``` input text using KeyCluster or EmbedRank.
For a German text run:
```
python src/extract_keyphrases_demo.py data/demo_confs/de_embedrank.ini demo_input.txt
```
or
```
python src/extract_keyphrases_demo.py data/demo_confs/de_keycluster.ini demo_input.txt
```

For an English text run:
```
python src/extract_keyphrases_demo.py data/demo_confs/en_embedrank.ini demo_input.txt
```
or
```
python src/extract_keyphrases_demo.py data/demo_confs/en_keycluster.ini demo_input.txt
```

The config files provide default settings for the unmodified versions of KeyCluster and EmbedRank. 
In order to test our modifications, refer to the config files and adjust the settings in there (especially paths 
for files like the word embedding models, etc.).
You can provide your own input text by editing ```demo_input.txt``` or providing your own file.