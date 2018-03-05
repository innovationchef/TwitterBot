#!/usr/bin/env python

import argparse
import contextlib
import logging
import os

import sqlite3

import gzip
import json

import tweety.cleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CLASSES
class ParseTweet:
    def __init__(self, tweet):
        self.tweet = tweet
        self.clean_tweet()

    def clean_tweet(self):
        text_cleaner = tweety.cleaner.TextCleaning()
        tweet_dict = json.loads(self.tweet)
        
        self._id  = tweet_dict['id']
        self.user  = tweet_dict['user']['id']
        self.retweet_count =  tweet_dict['retweet_count'] if type(tweet_dict['retweet_count']) is int else 100
        self.in_reply_to_status_id = tweet_dict['in_reply_to_status_id'] if type(tweet_dict['in_reply_to_status_id']) is int else None
        self.in_reply_to_user_id = tweet_dict['in_reply_to_user_id'] if type(tweet_dict['in_reply_to_user_id']) is int else None
        self.in_reply_to_screen_name = text_cleaner.preprocess(tweet_dict['in_reply_to_screen_name']) if type(tweet_dict['in_reply_to_screen_name']) is str else None
        self.created_at = str(tweet_dict['created_at']) if type(tweet_dict['created_at']) is str else None
        self.coordinates = str(tweet_dict['coordinates']) if type(tweet_dict['coordinates']) is str else None
        self.geo = str(tweet_dict['geo']) if type(tweet_dict['geo']) is str else None
        self.place = str(tweet_dict['place']) if type(tweet_dict['place']) is str else None
        self.truncated = tweet_dict['truncated']
        self.favorited = tweet_dict['favorited']
        self.retweeted = tweet_dict['retweeted']

        _text = tweet_dict['text'].encode('utf-8')
        if text_cleaner.preprocess(_text) is None: 
            self.text = None
        else:   
            self.text = text_cleaner.preprocess(_text)

class ParseUser:
    def __init__(self, tweet):
        self.tweet = tweet
        self.clean_user()

    def clean_user(self):
        tweet_dict = json.loads(self.tweet)['user']

        self.time_zone = str(tweet_dict['time_zone']) if type(tweet_dict['time_zone']) is str else None
        self._id  = tweet_dict['id']
        self.followers_count  = tweet_dict['followers_count']
        self.listed_count  = tweet_dict['listed_count']
        self.statuses_count  = tweet_dict['statuses_count']
        self.description = self.process(tweet_dict['description'])
        self.friends_count  = tweet_dict['friends_count']
        self.notifications = tweet_dict['notifications']
        self.show_all_inline_media = tweet_dict['show_all_inline_media']
        self.geo_enabled = tweet_dict['geo_enabled']
        self.name = self.process(tweet_dict['name'])
        self.lang = str(tweet_dict['lang'])
        self.favourites_count  = tweet_dict['favourites_count']
        self.screen_name = str(tweet_dict['screen_name'])
        self.created_at = str(tweet_dict['created_at'])
        self.contributors_enabled = tweet_dict['contributors_enabled']
        self.verified = tweet_dict['verified']
        self.protected = tweet_dict['protected']
        self.is_translator = tweet_dict['is_translator']
        self.location = self.process(tweet_dict['location'])

    def process(self, text):
        text_cleaner = tweety.cleaner.TextCleaning()
        if text is None:
            return None
        text = text.encode('utf-8')
        if text_cleaner.preprocess(text) is None: 
            return None
        else:   
            return text_cleaner.preprocess(text)


# FUNCTIONS
def readfiles(indir):
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            filename = indir + '/' + f
            for line in open(filename):
                yield line.rstrip('\n')


# MAIN
parser = argparse.ArgumentParser('Add the directory of raw Twitter data')

parser.add_argument('path_to_raw_directory', help='Path to the extracted directory used to crawl tweets.')

parser.add_argument('path_to_db', help='''Location to save extracted data.''')

parser.add_argument('--force-add', action='store_true', help='If true then directory is extracted even if they have already been extracted')

args = parser.parse_args()

if not os.path.exists(args.path_to_raw_directory):
    logger.error('directory %s does not exist', args.path_to_raw_directory)
    exit(1)

if not os.path.exists(args.path_to_db):
    logger.error('Database %s does not exist', args.path_to_db)
    exit(1)

cwd = os.getcwd()
indir = cwd + '/'+ args.path_to_raw_directory
db_path = args.path_to_db

with sqlite3.connect(db_path) as conn:
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row

    with contextlib.closing(conn.cursor()) as curs:
        i = 0
        for line in readfiles(indir):
            parsed_tweet = ParseTweet(line)
            parsed_user = ParseUser(line)
            curs.execute("INSERT OR REPLACE INTO tweets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (parsed_tweet._id,
                parsed_tweet.text, parsed_tweet.user, parsed_tweet.retweet_count, parsed_tweet.in_reply_to_status_id, parsed_tweet.in_reply_to_user_id,
                parsed_tweet.in_reply_to_screen_name, parsed_tweet.created_at, parsed_tweet.coordinates, parsed_tweet.geo, parsed_tweet.place, parsed_tweet.truncated,
                parsed_tweet.favorited, parsed_tweet.retweeted))
            curs.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (parsed_user.time_zone, parsed_user._id, 
                parsed_user.followers_count, parsed_user.listed_count, parsed_user.statuses_count, 
                parsed_user.description, parsed_user.friends_count, parsed_user.notifications, parsed_user.show_all_inline_media, parsed_user.geo_enabled,
                parsed_user.name, parsed_user.lang, parsed_user.favourites_count, parsed_user.screen_name, parsed_user.created_at,
                parsed_user.contributors_enabled, parsed_user.verified, parsed_user.protected, parsed_user.is_translator, parsed_user.location))
            logger.info('Dumped tweet ID %d in Database', parsed_tweet._id)
            i += 1
            
logger.info('Dumped %d Tweets in Database', i)
