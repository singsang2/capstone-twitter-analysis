import warnings
warnings.filterwarnings(action='ignore')
import tweepy
import pandas as pd
import sqlite3

import keys.keys as keys
from tweepy.streaming import StreamListener
import time
import json
import string
import re
# import spacy
# nlp = spacy.load('en_core_web_lg')
import ktrain
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# import requests
# import os
# import app
# import importlib

### Modify the follwoing query words before running! ###
QUERIES = ['Microsoft', 'Apple', 'Google', 'facebook','Comcast', 'AT&T', 'youtube']
           

# VADER MODEL
analyser = SentimentIntensityAnalyzer()
def vader_analyzer(tweet):
    score = analyser.polarity_scores(tweet)
    return score['compound'] 

# SQL Database
def create_table(cursor, conn):
    cursor.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, id TEXT, user TEXT, tweet TEXT, favorite INT, retweet INT, sentiment REAL)")
    conn.commit()


### Twitter Listener Class Modification to Meet Our Needs ###
class listener(StreamListener):
    def __init__(self, api, cursor, conn):
        self.cursor = cursor
        self.api = api
        self.conn = conn

    def on_data(self, data):
        try:
            # loads json data
            data = json.loads(data)
            if not data['retweeted'] and 'RT @' not in data['text']:
                tweet = data['text']
                time_ms = data['timestamp_ms']
                favorite = data['favorite_count'] 
                retweet = data['retweet_count']
                id_str = data['id_str']
                user_str = data['user']['id_str']

                # Sentiment Analysis *Change model if necessarity
                # sentiment = TextBlob(tweet).sentiment.polarity
                sentiment = vader_analyzer(tweet)
                # print(time_ms, tweet, sentiment)
                
                self.cursor.execute("INSERT INTO sentiment (unix, id, user, tweet, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (time_ms, id_str, user_str, tweet, favorite, retweet, sentiment))
                self.conn.commit()

        except KeyError as e:
            print(str(e))
            print('keyerror')
            time.sleep(1)
        return True
    
    def on_error(self, status):
        print(status)
        time.sleep(5)

# Timestamp
TIMESTAMP = str(datetime.date.today()).replace('-','')

# Connects to Twitter API via Tweepy
auth = tweepy.OAuthHandler(consumer_key=keys.CONSUMER_KEY, consumer_secret=keys.CONSUMER_SECRET) ### MUST SET YOUR OWN KEYS
auth.set_access_token(keys.ACCESS_KEY, keys.ACCESS_SECRET) ### MUST SET YOUR OWN KEYS
api = tweepy.API(auth, wait_on_rate_limit=True)

# Connects to sqlite3 database
# path=f'data/twitter_{TIMESTAMP}.db'
path=f'data/twitter_20210118.db'
conn = sqlite3.connect(path, check_same_thread=False, timeout=20)
cursor = conn.cursor()
create_table(cursor, conn)

while 1:
    try:
        print('Streaming Beginning...')
        print('Requested Queries: ', QUERIES)
        streamer = tweepy.Stream(auth, listener(api, cursor, conn))
        print('1')
        streamer.filter(track=QUERIES, languages=['en'], stall_warnings=True)
        print('2')
    except Exception as e:
        print(e)
        time.sleep(10)
        print("Reconnecting"*50)