
**Twitter Streaming NLP Analysis**

This project was inspired `sentdex`. (https://github.com/Sentdex/socialsentiment/)

# What is in this repository

**Jupyter Notebooks**
- `sentiment_models.ipynb`: contains training and validating of various models including BERT and TF-IDF models.

**Python Files**
- `twitter_stream.py`: connects to Twitter API and streamline tweets with various keywords
- `app.py`: Dash dashboard

**Folders**
- `images`: contains image files
- `models`: contains (1) BERT NLP and (2) spaCy TF-IF vectorization sentiment analysis models *(*BERT model excluded)*
- `data`: contains SQL databases pulled from Twitter using Tweepy *(only sample database is included)*
- `datasets`: contains datasets that were used to train various NLP sentiment analysis models
- `src`: contains useful codes that were used in creating models
- `keys`: contains Twitter API key information *(*files excluded)*

**How to use this project**

1. Add Twitter API information in `keys` directory, and make sure the PATH is correctly defined in `twitter_stream.py`.
2. Adjust any keywords or queries in `twitter_stream.py`.
3. Set up SQLITE3 database and run `twitter_stream.py`.
4. Run `app.py`.

<img src='images/dashboard_1.png'>
<img src='images/dashboard_2.png'>
<img src='images/dashboard_3.png'>
<img src='images/dashboard_4.png'>

# Introduction

## Business Case
As of January of 2020, there are approximately 145 million users on Twitter. 22% of Americans are on Twitter and 500 million tweets are sent each day globally. This is why many companies internationally use Twitter for marketing. In fact 80% of Twitter users have mentioned a brand in a tweet, and 77% of Twitter users feel more positive when their tweets have been replied by the mentioned brand [1].

We believe that Twitter is one of the platforms that provides people to share their opinion, evaluations, attitudes, and emotions about virtually anything including certain products freely, and for any companies, this is like a gold mine waiting to be mined for `opinions are central to almost all human activities and are key influencers of our behaviors` [2].


Sources:

[1] https://unsplash.com/photos/ulRlAm1ITMU

[2]. Bing Liu, https://www.morganclaypool.com/doi/abs/10.2200/s00416ed1v01y201204hlt016

## Goals
The goals of this project are to 

    [1] stream Twitter with on various topics (ex. Microsoft, Starbucks, Google, etc.) and
    [2] effectively implement various NLP models (including TextBlob, BERT, and TF-IDF models) to classify tweets
    [3] to flag the user for any strongly negative tweets on a Dashboard to effectively respond to them.

    

# Conclusion

BERT model has accuracy of 84%, however due to its computing time,Textblob was used as an initial model to classify polarity of each tweet. The BERT model was used to confirm any strongly negative sentiment tweets classified by Textblob.



# Future Work

1. Database structure
    - As the database size increases, it might get too much for sqlite3 to handle. So, we might consider saving 
2. Reply feature
    - It would be nice if we could add a feature where you could reply to any of tweets shown in the dashboard without going to twitter page.
3. Multiple keywords
    - It would be nice if multiple keywords can be analyzed and followed at a given time.
4. Table Editing Mode
    - Modify flagged tweets in the dashboard so that a user can classify flagged tweets as 'resolved', 'false negative', or 'other' for further customer service / data analysis.
5. Further analysis on both positive and negative tweets
    - Find any correlation between tweet trends with how the company is doing to help with future direction of a company.


```python

```
