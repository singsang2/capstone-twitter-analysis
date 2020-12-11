import warnings
warnings.filterwarnings(action='ignore')
import tweepy
import pandas as pd
import sqlite3
from textblob import TextBlob
import keys.keys as keys
from tweepy.streaming import StreamListener
import time
import json
import string
import re
import spacy
nlp = spacy.load('en_core_web_lg')
import ktrain
import datetime
# import requests
# import os
# import app
# import importlib

### Modify the follwoing query words before running! ###
QUERIES = ['Microsoft OR to:Microsoft OR #Microsoft OR @Microsoft, entity:"Microsoft"', 
           'Apple, ios, iphone, ipad, imac, mac, macbook, macair, airpod, entity:"Apple"', 
           'Google, entity:"Google"', 
           'Chickfila, (Chick fil), (Chick fila), entity:"Chickfila"', 
           'BurgerKing, entity:"BurgerKing', 
           'McDonalds, entity:"McDonalds"', 
           'Coca, coke, entity:"Coca cola", entity:"Pepsi"', 'chicken', 
           'Starbucks, entity:"Starbucks"', 'entity:"youtube"',
           'Hyundai, entity:"Hyundai"', 'Bentz: entity:"Bentz"', "Mercedes", 'toyota, entity:"toyota"', 'kia',
           'Samsung, entity:"Samsung"', 'Sony, entity:"Sony"', 'LG', 
           'Azure', 'Amazon, entity:"Amazon"', 'yahoo', 'AWS', 
           'Surface', 'macbook', 'macpro', 'imac', 'iphone', 'android', 'galaxy',
           'Windows', 'ios', 'Xbox',
           'Fedex, UPS, USPS, delivery, entity:"Fedex", entity:"UPS", entity:"UPSP"'
           'instagram', 'tiktok', 'facebook, entity:"facebook"', 'PS5', 'Zoom', 'Telegram', 'Tesla, entity:"Tesla"', 'SpaceX, entity:"SpaceX"']


### spaCy tokenizer ###
def clean_text(text, stopwords=False, tweet=True):
    """
    Cleans and tokenizes tweet text data.
    Args:
        text (str): tweet text data
        
        stopwords (bool): True if stopwords needs to be removed
        
        tweet (bool): True if text data are tweets.
    
    Returns:
        tokens (array): Array of tokenized words from given text.
    """

    if tweet:
        text = re.sub(r'@\S+', '', text) # Gets rid of any mentions
        text = re.sub(r'RT\S+', '', text) # Gets rid of any retweets
        text = re.sub(r'#', '', text) # Gets rid of hashtag sign
        text = re.sub(r'https?:\/\/\S+', '', text) # Gets rid of any links
        text = re.sub(r'[0-9]+.?[0-9]+', '', text) # Gets rid of X.X where X are numbers
        text = re.sub(r'#?(sx|Sx|SX)\S+', '', text) # Gets rid common mentions
        text = re.sub(r'(&quot;|&Quot;)', '', text) # Gets rid of quotes    
        text = re.sub(r'(&amp;|&Amp;)', '', text) # Gets rid of quotes
        text = re.sub(r'link', '', text) # Gets rid of quotes
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-': # if token is not a pronoun
            temp_token = token.lemma_.lower().strip()
        else:
            temp_token = token.lower_
        tokens.append(temp_token)
    
    if stopwords:
        # tokens_stopped = [token for token in tokens if token not in stopwords_list and len(token)>2]
        pass
    else:
        tokens_stopped = [token for token in tokens if len(token)>2]
    
    return ' '.join(tokens_stopped)

### Twitter Listener Class Modification to Meet Our Needs ###
class listener(StreamListener):
    def __init__(self, cursor, conn, predictor, neg_threshold):
        self.cursor = cursor
        self.conn = conn
        self.predictor = predictor
        self.neg_threshold = neg_threshold

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

                clean = clean_text(data['text'])
                # Sentiment Analysis *Change model if necessarity
                sentiment = TextBlob(tweet).sentiment.polarity
                print(time_ms, tweet, sentiment)
                
                self.cursor.execute("INSERT INTO sentiment (unix, id, user, tweet, clean, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment))
                self.conn.commit()

                if sentiment < self.neg_threshold:  ### This NEG_THRESH value can be adjusted by the user
                    proba = self.predictor.predict_proba([tweet])[0]
                    # print('BERT EXCUTED!')
                    if proba[0] > proba[1]:
                        self.cursor.execute("INSERT INTO flag (unix, id, user, tweet, clean, favorite, retweet, sentiment, dealt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment*proba[0], 0))
                        self.conn.commit()
                        print('Stream: FLAGGGGGEED!')
        except KeyError as e:
            print(str(e))
            time.sleep(2)
        return True
    
    def on_error(self, status):
        print(status)
        time.sleep(5)

class streamTwitter():
    def __init__(self, current_keywords, timestamp, model_path='models/BERT_2'):
        self.current_keywords = current_keywords
        self.timestamp = timestamp
        self.neg_threshold = -0.4

        # Loads BERT model
        self.predictor = ktrain.load_predictor(model_path)
        
        # Connects to Twitter API via Tweepy
        self.auth = tweepy.OAuthHandler(consumer_key=keys.CONSUMER_KEY, consumer_secret=keys.CONSUMER_SECRET) ### MUST SET YOUR OWN KEYS
        # self.auth = tweepy.AppAuthHandler(consumer_key=keys.CONSUMER_KEY, consumer_secret=keys.CONSUMER_SECRET)
        self.auth.set_access_token(keys.ACCESS_KEY, keys.ACCESS_SECRET) ### MUST SET YOUR OWN KEYS
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

        # Connects to sqlite3 database
        path=f'data/twitter_{self.timestamp}.db'
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=20)
        self.cursor = self.conn.cursor()
        self.create_table()
        self.create_flag_table()

    ### SQL Connection / creating tables ###
    def create_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL)")
        self.conn.commit()

    def create_flag_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS flag(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL, dealt INT)")
        self.conn.commit()

    def stream(self, languages=['en']):
        """
        Runs streamining from Twitter.
        Args:
            q (list): list of strings containing queries for Tweet filter

            languages (list): list of strings containing languages for tweet filter
        """
        print('Streaming Beginning...')
        print('Requested Queries: ', self.current_keywords)
        self.twitterStream = tweepy.Stream(self.auth, listener(self.cursor, self.conn, self.predictor, self.neg_threshold))
        self.twitterStream.filter(track=self.current_keywords, languages=languages)
        # twitterStream.filter(track=['a', 'the', 'i', 'you', 'to'], languages=['en'])

    # def search_tweet(self, search_words, max_len=5000):
    #     tweets = tweepy.Cursor(self.api.search, q=[f'{word}' for word in search_words],     #-filter:retweets
    #                            count=100, result_type='recent', 
    #                            lang='en').items(max_len)
    #     for tweet in tweets:
    #         data = tweet._json
    #         tweet = data['text']
    #         date_time = data['created_at']
    #         time_ms = time.mktime(datetime.datetime.strptime(date_time,'%a %b %d %H:%M:%S +0000 %Y').timetuple())*1000
    #         favorite = data['favorite_count'] 
    #         retweet = data['retweet_count']
    #         id_str = data['id_str'] 
    #         user_str = data['user']['id_str']

    #         clean = clean_text(data['text'])
    #         # Sentiment Analysis *Change model if necessarity
    #         sentiment = TextBlob(tweet).sentiment.polarity
    #         # print(time_ms, tweet, sentiment)

    #         self.cursor.execute("INSERT INTO sentiment (unix, id, user, tweet, clean, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    #                       (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment))
    #         self.conn.commit()

    #         if sentiment < self.neg_threshold:  ### This NEG_THRESH value can be adjusted by the user
    #             proba = self.predictor.predict_proba([tweet])[0]
    #             # print('BERT EXCUTED!')
    #             if proba[0] > proba[1]:
    #                 self.cursor.execute("INSERT INTO flag (unix, id, user, tweet, clean, favorite, retweet, sentiment, dealt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    #                     (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment*proba[0], 0))
    #                 self.conn.commit()
    #                 print('Search: FLAGGGGGEED!')

    def select_search_words(self, keywords):
        new_batch_words = []
        for word in keywords:
            if word.lower() not in self.current_keywords:
                self.current_keywords.append(word.lower())
                new_batch_words.append(word)
        return new_batch_words

    def disconnect(self):
        self.twitterStream.disconnect()

    def update_stream(self, keywords):
        search_words = self.select_search_words(keywords)
        if len(search_words)>0:
            print('New Added keywords: ', search_words)
            start = time.time()
            self.search_tweet(search_words)
            print(f'search done. Running time: {time.time()-start}s')

            # self.twitterStream().disconnect()
            print(f'Begin Streaming. Keywords: {self.current_keywords}')
            self.stream()
        # except Exception as e:
        #     print('DISCONNECTED')
        #     print(e)
        #     time.sleep(10)
        #     print('*'*100) 
        #     print('RECONNECTING '*20)
        #     print('*'*100)
TIMESTAMP = str(datetime.date.today()).replace('-','')
while 1:
    try:
        streamer = streamTwitter(current_keywords=QUERIES, timestamp=TIMESTAMP)
        streamer.stream()
    except Exception as e:
        print(e)
        time.sleep(10)
        print("Reconnecting"*50)