import logging

import canonicaljson
import hashlib
import json
import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Index:
    def __init__(self, sentence):
        self.sentence = sentence
        self.index()

    def index(self):
        headers = {'Content-type': 'application/json'}
        solr_json = self._create_solr_json()

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

    def _create_solr_json(self):
        json_struct = {}
        json_struct['sentence'] = self.sentence
        return json_struct
