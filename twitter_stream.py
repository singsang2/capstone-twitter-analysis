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
                
                c.execute("INSERT INTO sentiment (unix, id, user, tweet, clean, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment))
                conn.commit()

                if sentiment < NEG_THRESH:  ### This NEG_THRESH value can be adjusted by the user
                    proba = predictor.predict_proba([tweet])[0]
                    print('BERT EXCUTED!')
                    if proba[0] > proba[1]:
                        c.execute("INSERT INTO flag (unix, id, user, tweet, clean, favorite, retweet, sentiment, dealt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment*proba[0], 0))
                        conn.commit()
                        print('FLAGGGGGEED!')
        except KeyError as e:
            print(str(e))
            time.sleep(2)
        return True
    
    def on_error(self, status):
        print(status)
        time.sleep(5)

class stream_twitter():
    def __init__(self, current_keywords, timestamp, model_path='models/BERT_2'):
        self.keywords = keywords
        self.timestamp = timestamp
        self.neg_threshold = -0.4

        # Loads BERT model
        self.predictor = ktrain.load_predictor(MODEL_PATH)
        
        # Connects to Twitter API via Tweepy
        self.auth = tweepy.OAuthHandler(consumer_key=keys.CONSUMER_KEY, consumer_secret=keys.CONSUMER_SECRET) ### MUST SET YOUR OWN KEYS
        self.auth.set_access_token(keys.ACCESS_KEY, keys.ACCESS_SECRET) ### MUST SET YOUR OWN KEYS
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

        # Connects to sqlite3 database
        path=f'data/twitter_{TIMESTAMP}.db'
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=20)
        self.cursor = conn.cursor()
        self.create_table()
        self.create_flag_table()

    ### SQL Connection / creating tables ###
    def create_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL)")
        self.conn.commit()

    def create_flag_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS flag(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL, dealt INT)")
        self.conn.commit()

    def stream(self, q, languages=['en']):
        """
        Runs streamining from Twitter.
        Args:
            q (list): list of strings containing queries for Tweet filter

            languages (list): list of strings containing languages for tweet filter
        """
        print('Streaming Beginning...')
        print('Requested Queries: ', q)
        twitterStream = tweepy.Stream(auth, listener())
        twitterStream.filter(track=q, languages=languages)
        # twitterStream.filter(track=['a', 'the', 'i', 'you', 'to'], languages=['en'])

    def search_tweet(self, keyword, max_len=500):
        tweets = tweepy.Cursor(api.search, q=[f'{keyword} -filter:retweets', f'to:{keyword}'], count=100, result_type='recent', lang='en').items(max_len)
        for tweet in tweets:
            data = tweet._json
            tweet = data['text']
            time = data['creted_at']
            time_ms = time.mktime(datetime.datetime.strptime(time,'%a %b %d %H:%M:%S +0000 %Y').timetuple())
            favorite = data['favorite_count'] 
            retweet = data['retweet_count']
            id_str = data['id_str']
            user_str = data['user']['id_str']

            clean = clean_text(data['text'])
            # Sentiment Analysis *Change model if necessarity
            sentiment = TextBlob(tweet).sentiment.polarity
            # print(time_ms, tweet, sentiment)

            c.execute("INSERT INTO sentiment (unix, id, user, tweet, clean, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment))
            conn.commit()

            if sentiment < NEG_THRESH:  ### This NEG_THRESH value can be adjusted by the user
                proba = predictor.predict_proba([tweet])[0]
                print('BERT EXCUTED!')
                if proba[0] > proba[1]:
                    c.execute("INSERT INTO flag (unix, id, user, tweet, clean, favorite, retweet, sentiment, dealt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment*proba[0], 0))
                    conn.commit()
                    print('FLAGGGGGEED!')

    def select_search_words(self, keywords):
        new_batch_words = []
        for word in keywords:
            if word.lower() not in current_keywords:
                current_keywords.append(word.lower())
                new_batch_words.append(word)
        return new_batch_words

    def start_stream(self, keywords):
        search_words = self.select_search_words(keywords)
        if len(search_words)>0:
            self.search_tweet(search_words)
        while 1:
            try:
                self.stream(q=current_keywords)
            except Exception as e:
                print('DISCONNECTED')
                print(e)
                time.sleep(10)
                print('*'*100) 
                print('RECONNECTING '*20)
                print('*'*100)

