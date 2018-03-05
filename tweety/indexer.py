import logging

import canonicaljson
import hashlib
import json
import requests

from nltk import word_tokenize
import os

import nltk

logger = logging.getLogger(__name__)

class LoadCorpus:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def readfiles(self):
        indir = self.data_directory
        for root, dirs, filenames in os.walk(indir):
            for f in filenames:
                filename = indir + '/' + f
                for line in open(filename):
                    yield line.rstrip('\n')        

class IndexCorpus:
    def __init__(self, sentence):
        self.sentence = sentence
        self.readlines()
        logger.info('check')
        
    def readlines(self):
        sentence_tokens = word_tokenize(self.sentence.lower())
        self.trigrams = nltk.ngrams(sentence_tokens,3) 

    def index(self):
        for trigram in self.trigrams:
            
            self._index(trigram)


    def _index(self, trigram):
        headers = {'Content-type': 'application/json'}
        solr_json = self._create_solr_json(trigram)

        solr_json['id'] = hashlib.sha256(canonicaljson.encode_canonical_json(solr_json)).hexdigest()

        config = {
                    'post_to_solr': True,
                    'solr_json_doc_update_url': 'http://localhost:8983/solr/twitter/update/json/docs',
                    'solr_query_url': 'http://localhost:8983/solr/twitter/select'
                }


        if config['post_to_solr']:
            r = requests.get(config['solr_query_url'] + '?q=id:' + solr_json['id'])
            if r.status_code != 200:
                logger.error('Could not post to Solr: %s', r.text)
            r_json = json.loads(r.text)
            num_found = int(r_json['response']['numFound'])
            if num_found > 0:
                logger.info('Already indexed')
                return

            logger.debug('Posting %s', solr_json)

            r = requests.post(config['solr_json_doc_update_url'] + '?commit=true', json=solr_json, headers=headers)
            if r.status_code != 200:
                logger.error('Could not post to Solr: %s', r.text)

    def _create_solr_json(self, trigram):
        json_struct = {}
        try: 
            json_struct['word1'], json_struct['word2'], json_struct['word3'] = trigram
            json_struct['ufreq'], json_struct['bfreq'], json_struct['tfreq'] = 1, 1, 1
        except StopIteration: pass
        return json_struct

if __name__ == "__main__":
    loadcorpus = LoadCorpus('processed_data')
    for line in loadcorpus.readfiles():
        print line
        IndexCorpus(line).index()
  