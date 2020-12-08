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

auth = tweepy.OAuthHandler(consumer_key=keys.CONSUMER_KEY, consumer_secret=keys.CONSUMER_SECRET)
auth.set_access_token(keys.ACCESS_KEY, keys.ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

conn = sqlite3.connect('data/twitter_2.db')
c = conn.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, id TEXT, user TEXT, tweet TEXT, clean TEXT, favorite INT, retweet INT, sentiment REAL)")
    conn.commit()
create_table()

# spaCy tokenizer
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


class listener(StreamListener):
    def on_data(self, data):
        try:
            # loads json data
            data = json.loads(data)
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
            
            c.execute("INSERT INTO sentiment (unix, id, user, tweet, clean, favorite, retweet, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (time_ms, id_str, user_str, tweet, clean, favorite, retweet, sentiment))
            conn.commit()
            
        except KeyError as e:
            print(str(e))
            time.sleep(5)
        return True
    
    def on_error(self, status):
        print(status)
        time.sleep(5)

twitterStream = tweepy.Stream(auth, listener())
twitterStream.filter(track=['Microsoft', 'to:Microsoft', '#Microsoft', '@Microsoft',
                            'to:Windows', '@Windows', '#Windows', 'microsoft windows'
                            'to:Xbox', '@Xbox' ,'#Xbox', 'Xbox filter:retweets',
                            'Microsoft surface', 'Microsoft office', 'microsoft azure', 'microsoft teams', 'microsoft cloud',
                            'to:LinkedIn', '@LinkedIn', '#LinkedIn', 'LinkedIn'
                            'to:Apple', '#Apple', '@Apple', 'Apple',
                            'to:Google', '#Google', "@Google", 'Google',
                            'Honda', 'Kia','Toyota', 'Ford', 'Hyundai', 'BMW', 'MercedesBenz', 'Tesla',
                            'Starbucks', 'ipad', 'itunes', 'ios', 'airpod',
                            'HomeDepot', 'Azure', 'AWS'
                            'Facebook', 'instagram', 'Snapchat', 
                            'Samsung', 'Sony', 'LG', 
                            'McDonalds', 'BurgerKing', 'ChickfilA', 'PopeyesChicken',
                            'PlayStation', 'PS5'], languages=['en'])
# twitterStream.filter(track=['a', 'the', 'i', 'you', 'to'], languages=['en'])