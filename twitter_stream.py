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

### Constants ###
# Timestamp for today's date
TIMESTAMP = str(datetime.date.today()).replace('-','')
# Flag Sentiment Threshold
NEG_THRESH = -0.4    # any tweets that have less than this value by TextBlob will be re-evaluated by BERT

### SQL Connection / creating tables ###
def create_table(c, conn):
    c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL)")
    conn.commit()

def create_flag_table(c, conn):
    c.execute("CREATE TABLE IF NOT EXISTS flag(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL, dealt INT)")
    conn.commit()

def connect_sqlite(timestamp):
    path=f'data/twitter_{timestamp}.db'
    conn = sqlite3.connect(path, check_same_thread=False, timeout=20)
    c = conn.cursor()

    return c, conn

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
    def __init__(self, predictor, cursor, conn):
        self.predictor = predictor
        self.cursor = cursor
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

                clean = clean_text(data['text'])
                # Sentiment Analysis *Change model if necessarity
                sentiment = TextBlob(tweet).sentiment.polarity
                # print(time_ms, tweet, sentiment)
                
                self.cursor.execute("INSERT INTO sentiment (unix, id, user, tweet, clean, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment))
                self.conn.commit()

                if sentiment < NEG_THRESH:  ### This NEG_THRESH value can be adjusted by the user
                    proba = self.predictor.predict_proba([tweet])[0]
                    print('BERT EXCUTED!')
                    if proba[0] > proba[1]:
                        self.cursor.execute("INSERT INTO flag (unix, id, user, tweet, clean, favorite, retweet, sentiment, dealt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment*proba[0], 0))
                        self.conn.commit()
                        print('FLAGGGGGEED!')
        except KeyError as e:
            print(str(e))
            time.sleep(2)
        return True
    
    def on_error(self, status):
        print(status)
        time.sleep(5)

def stream(auth, predictor, c, conn, q, languages=['en']):
    print('Streaming Beginning...')
    print('Requested Queries: ', q)
    twitterStream = tweepy.Stream(auth, listener(predictor, c, conn))
    twitterStream.filter(track=q, languages=languages)
    # twitterStream.filter(track=['a', 'the', 'i', 'you', 'to'], languages=['en'])

def run_stream(timestamp=TIMESTAMP, q=['a', 'the', 'i', 'you', 'to']):
    # Loads BERT model
    MODEL_PATH = 'models/BERT_2' ### CHANGE THE PATH ACCORDINGLY
    predictor = ktrain.load_predictor(MODEL_PATH)

    # Connects to Twitter API via Tweepy
    auth = tweepy.OAuthHandler(consumer_key=keys.CONSUMER_KEY, consumer_secret=keys.CONSUMER_SECRET) ### MUST SET YOUR OWN KEYS
    auth.set_access_token(keys.ACCESS_KEY, keys.ACCESS_SECRET) ### MUST SET YOUR OWN KEYS
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Connects to sqlite3 database
    c, conn = connect_sqlite(timestamp)
    create_table(c, conn)
    create_flag_table(c, conn)
    while 1:
        try:
            stream(auth, predictor, c, conn, q=q)
        except Exception as e:
            print('DISCONNECTED')
            print(e)
            time.sleep(10)
            print('*'*100) 
            print('RECONNECTING '*20)
            print('*'*100)

run_stream(TIMESTAMP)