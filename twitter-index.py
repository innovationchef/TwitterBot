#!/usr/bin/env python

import os
import logging
import argparse
import sqlite3
import contextlib

import tweety.indexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# MAIN
parser = argparse.ArgumentParser('Index extracted tweets into Solr.')
parser.add_argument('path_to_db', help='Path to the database used to store extracted tweets.')
parser.add_argument('-s', '--solr-core-url', nargs='?', help='URL to solr endpoint')
args = parser.parse_args()

if args.solr_core_url is None:
    endpoint = 'http://localhost:8983/solr/twitter/'
else:
    endpoint = str(args.solr_core_url)
    logger.info('Indexing at Solr core %s', args.solr_core_url)

if not os.path.exists(args.path_to_db):
    logger.error('Extracted database %s does not exist', args.path_to_db)
    exit(1)

config = {
    'post_to_solr': True,
    'solr_json_doc_update_url': endpoint + 'update/json/docs',
    'solr_query_url': endpoint +'select'
}

with sqlite3.connect(args.path_to_db) as conn:
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row

    with contextlib.closing(conn.cursor()) as curs:
        curs.execute('SELECT * FROM tweets')
        while True:
            row = curs.fetchone()    
            if row == None:
                break
            if row['tweet'] == None:
                pass
            else:
                tweety.indexer.Index(row['tweet'])
                logger.info('Indexing tweet %s', row['id'])
