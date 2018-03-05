#!/usr/bin/env python

import contextlib
import sqlite3
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#MAIN
parser = argparse.ArgumentParser('Setup a database to store extracted data')
parser.add_argument('path_to_db', help='Path to the database used to store extracted information, e.g data/extracted.db')

args = parser.parse_args()

# TWEET DATA
# id INTEGER PRIMARY KEY                    # int
# tweet TEXT                                # Text
# user INTEGER                              # Extracted Integer Id
# retweet_count INTEGER                     # int or unicode "100+"
# in_reply_to_status_id INTEGER             # None or int
# in_reply_to_user_id INTEGER               # None or Int
# in_reply_to_screen_name TEXT              # none or text
# created_at TEXT                           # Date in Text Format
# coordinates TEXT                          # none or {u'type': u'Point', u'coordinates': [51.410388, -0.072143]}
# geo TEXT                                  # none or {u'type': u'Point', u'coordinates': [51.410388, -0.072143]}
# place TEXT                                # none or dict --- {u'name': u'University of Texas - Austin', u'url': u'http://api.twitter.com/1/geo/id/b94384e044bfacc4.json', u'country': u'United States', u'place_type': u'neighborhood', u'bounding_box': {u'type': u'Polygon', u'coordinates': [[[-97.74218304, 30.27870801], [-97.72118208, 30.27870801], [-97.72118208, 30.293307], [-97.74218304, 30.293307]]]}, u'full_name': u'University of Texas - Austin, Austin', u'attributes': {\}, u'country_code': u'US', u'id': u'b94384e044bfacc4'}
# truncated BOOLEAN                         # All False
# favorited BOOLEAN                         # All False
# retweeted BOOLEAN                         # All False

# USER DATA
# time_zone TEXT                            # Text or none
# id INTEGER                                # Int
# followers_count INTEGER                   # Int
# listed_count INTEGER                      # Int
# statuses_count INTEGER                    # Int
# description TEXT                          # Text
# friends_count INTEGER                     # Int
# notifications BOOLEAN                     # None or False
# show_all_inline_media BOOLEAN             # True and False
# geo_enabled BOOLEAN                       # True and False
# name TEXT                                 # Text
# lang TEXT                                 # en etc
# favourites_count INT                      # Int
# screen_name TEXT                          # Text
# created_at TEXT                           # Text
# contributors_enabled BOOLEAN              # All FAlse 
# verified BOOLEAN                          # All False
# protected BOOLEAN                         # All False
# is_translator BOOLEAN                     # All False
# location TEXT                             # random text or no text

with sqlite3.connect(args.path_to_db) as conn:
    with contextlib.closing(conn.cursor()) as curs:
        curs.execute("""
                    CREATE TABLE IF NOT EXISTS tweets (
                        id INTEGER PRIMARY KEY,
                        tweet TEXT,
                        user INTEGER,
                        retweet_count INTEGER,
                        in_reply_to_status_id INTEGER,
                        in_reply_to_user_id INTEGER,
                        in_reply_to_screen_name TEXT,
                        created_at TEXT,
                        coordinates TEXT,
                        geo TEXT,
                        place TEXT,
                        truncated BOOLEAN,
                        favorited BOOLEAN,
                        retweeted BOOLEAN,
                        UNIQUE(id))
                    """)
        curs.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        time_zone TEXT,
                        id INTEGER PRIMARY KEY,
                        followers_count INTEGER,
                        listed_count INTEGER,
                        statuses_count INTEGER,
                        description TEXT,
                        friends_count INTEGER,
                        notifications BOOLEAN,
                        show_all_inline_media BOOLEAN,
                        geo_enabled BOOLEAN,
                        name TEXT,
                        lang TEXT,
                        favourites_count INT,
                        screen_name TEXT,
                        created_at TEXT,
                        contributors_enabled BOOLEAN,
                        verified BOOLEAN,
                        protected BOOLEAN,
                        is_translator BOOLEAN,
                        location TEXT,
                        UNIQUE(id))
                    """)
        conn.commit()
logger.info('Created Database at %s', args.path_to_db)
