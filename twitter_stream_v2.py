#### Work in Progress ####
import sqlite3
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import requests
import os
import json
import keys.keys as keys
import time


# VADER MODEL
analyser = SentimentIntensityAnalyzer()
def vader_analyzer(tweet):
    score = analyser.polarity_scores(tweet)
    return score['compound'] 

# SQL Database
def create_table(cursor, conn):
    cursor.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, id TEXT, user TEXT, tweet TEXT, tag TEXT, sentiment REAL)")
    conn.commit()


### Twitter Filtered Stream ###
### SOURCE: https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/master/Filtered-Stream/filtered_stream.py
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def get_rules(headers, bearer_token):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(headers, bearer_token, rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))


def set_rules(headers, delete, bearer_token):
    # You can adjust the rules if needed
    sample_rules = [
        {"value": "entity:Microsoft lang:en -is:retweet", "tag": "Microsoft"},
        {"value": "entity:Apple lang:en -is:retweet", "tag": "Apple"}
    ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))


def get_stream(headers, set, bearer_token, cursor, conn):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream?tweet.fields=created_at&expansions=author_id&user.fields=created_at", headers=headers, stream=True,
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    for response_line in response.iter_lines():
        if response_line:
            data = json.loads(response_line)['data']
            tag = json.loads(response_line)['matching_rules']
            # print(data.keys(), tag)
            tweet = data['text']
            time_ms = datetime.datetime.strptime(data['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()*1000
            id_str = data['id']
            user_str = data['author_id']
            print(tweet)

            sentiment = vader_analyzer(tweet)
                # print(time_ms, tweet, sentiment)
            # print('sentiment ', sentiment)
            # print(type(time_ms), type(id_str), type(user_str), type(tweet), type(tag[0]['tag']), type(sentiment))
            cursor.execute("INSERT INTO sentiment (unix, id, user, tweet, tag, sentiment) VALUES (?, ?, ?, ?, ?, ?)",
                (time_ms, id_str, user_str, tweet, tag[0]['tag'], sentiment))
            conn.commit()

def main():
    # Timestamp
    TIMESTAMP = str(datetime.date.today()).replace('-','')

    # Connects to sqlite3 database
    path=f'data/twitter_{TIMESTAMP}_v2.db'
    conn = sqlite3.connect(path, check_same_thread=False, timeout=20)
    cursor = conn.cursor()
    create_table(cursor, conn)

    bearer_token = keys.BEARER_TOKEN #os.environ.get("BEARER_TOKEN")
    headers = create_headers(bearer_token)
    rules = get_rules(headers, bearer_token)
    delete = delete_all_rules(headers, bearer_token, rules)
    new_rules = set_rules(headers, delete, bearer_token)
    try:
        get_stream(headers, new_rules, bearer_token, cursor, conn)
    except Exception as e:
        print(e)
        print('Reconnecting')
        time.sleep(5)

if __name__ == "__main__":

    main()

####