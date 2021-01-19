import warnings
warnings.filterwarnings(action='ignore')
import datetime
import sqlite3

import re
import spacy
nlp = spacy.load('en_core_web_lg')
import ktrain
import pandas as pd
import time
from collections import Counter
import pickle

BAG_OF_WORDS = Counter()

# Load BERT model
MODEL_PATH = 'models/BERT_2'
predictor = ktrain.load_predictor(MODEL_PATH)

# Timestamp for today's date
TIMESTAMP = str(datetime.date.today()).replace('-','')

# streamer = twitter_stream.streamTwitter([], TIMESTAMP)

### SQL Connection ###
# conn = sqlite3.connect(f'data/twitter_{TIMESTAMP}.db', check_same_thread=False, timeout=25)
conn = sqlite3.connect(f'data/twitter_20210118.db', check_same_thread=False, timeout=25)
c = conn.cursor()


# SQL Database
def create_table(cursor, conn):
    cursor.execute("CREATE TABLE IF NOT EXISTS flagged(unix REAL, id TEXT, clean TEXT, vader REAL, bert REAL, mood TEXT, sentiment REAL)")
    conn.commit()

def create_clean_table(cursor, conn):
    cursor.execute("CREATE TABLE IF NOT EXISTS clean(unix REAL, id TEXT, clean TEXT)")
    conn.commit()

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
#         text = re.sub(r'RT\S+', '', text) # Gets rid of any retweets
        text = re.sub(r'#', '', text) # Gets rid of hashtag sign
        text = re.sub(r'https?:\/\/\S+', '', text) # Gets rid of any links
#         text = re.sub(r'[0-9]+.?[0-9]+', '', text) # Gets rid of X.X where X are numbers
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

    BAG_OF_WORDS.update(tokens_stopped)
    with open('data/bag_of_words', 'wb') as f:
        pickle.dump(BAG_OF_WORDS, f)

    return ' '.join(tokens_stopped)

def main():
    create_table(c, conn)
    create_clean_table(c, conn)
    clean_time = datetime.datetime.timestamp(datetime.datetime.now())*1000
    analyze_time = clean_time
    time.sleep(10)
    while 1:
        clean_df = pd.read_sql(f"""SELECT * FROM sentiment 
                                    WHERE unix > {clean_time} AND sentiment NOT BETWEEN -0.1 AND 0.1
                                    ORDER BY unix ASC""", conn)
        if len(clean_df) > 0:
            print(f'There are {len(clean_df)} new tweets to clean!')
            new_clean_df = pd.DataFrame(columns=['unix', 'id', 'clean'])
            new_clean_df['unix'] = clean_df['unix']
            new_clean_df['id'] = clean_df['id']
            new_clean_df['clean'] = clean_df['tweet'].apply(clean_text)

            new_clean_df.to_sql('clean', conn, if_exists='append', index=False)
            print(clean_df.iloc[-1]['unix'])
            clean_time = clean_df.iloc[-1]['unix']

            temp_df = pd.read_sql(f"""SELECT s.unix, s.id, s.sentiment, c.clean FROM sentiment s
                                    JOIN clean c
                                    ON s.id = c.id
                                    WHERE s.unix > {analyze_time} AND sentiment NOT BETWEEN -0.69 and 0.79
                                    ORDER BY s.unix ASC""", conn)
            if len(temp_df) > 0:
                print(f'There are {len(temp_df)} new tweets to analyze!')
                analyze_time = temp_df.iloc[-1]['unix']

                new_df = pd.DataFrame(columns=['unix', 'id', 'clean', 'VADER', 'BERT', 'mood', 'sentiment'])

                new_df['clean'] = temp_df['clean']
                new_df['unix'] = temp_df['unix']
                new_df['id'] = temp_df['id']
                new_df['VADER'] = temp_df['sentiment']

                proba = predictor.predict_proba(list(new_df['clean']))

                bert = []
                mood = []
                senti = []
                for i, prob in enumerate(proba):
                    if prob[0] > prob[1]:
                        bert.append(prob[0])
                        if new_df['VADER'][i] < 0:
                            senti.append(new_df['VADER'][i]*prob[0])
                            mood.append('Negative')
                        else:
                            senti.append(0)
                            mood.append('?')
                    else:
                        bert.append(prob[1])
                        if new_df['VADER'][i] > 0:
                            senti.append(new_df['VADER'][i]*prob[1])
                            mood.append('Positive')
                        else:
                            senti.append(0)
                            mood.append('?')

                new_df['BERT'] = bert
                new_df['mood'] = mood
                new_df['sentiment'] = senti

                new_df.to_sql('flagged', conn, if_exists='append', index=False)
            else:
                print('No new tweets to analyze')
                analyze_time = clean_time
        else:
            print('No new tweets to analyze')
            time.sleep(5)
    
    


if __name__ == "__main__":
    main()