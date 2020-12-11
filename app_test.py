import warnings
warnings.filterwarnings(action='ignore')
import dash_table
import pandas as pd
import sqlite3 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly
import plotly.graph_objs as go
import plotly.express as px
import random

import pandas_datareader.data as web
import datetime

import string
import re
from PIL import Image
from wordcloud import WordCloud
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from spacy.lang.en.stop_words import STOP_WORDS

import base64
import time
import datetime
import urllib
import urllib.parse
# import twitter_stream

# Inspired by https://github.com/Sentdex/socialsentiment

# Keywords used for quries in tweet streaming
global CURRENT_KEYWORDS
CURRENT_KEYWORDS = []


global refresh_time
refresh_time = 5

# Timestamp for today's date
TIMESTAMP = str(datetime.date.today()).replace('-','')

# streamer = twitter_stream.streamTwitter([], TIMESTAMP)

### SQL Connection ###
conn = sqlite3.connect(f'data/twitter_{TIMESTAMP}.db', check_same_thread=False, timeout=25)
c = conn.cursor()

POS_THRESH = 0.3
NEG_THRESH = -0.3

#### STYLE ####
colors = {'background': '#111111', 
          'text':'#7FDBFF',
          'table-text':'#EAEAEA',
          'sentiment-graph-0': '#00FFE8',
          'sentiment-graph-1': '#FF4200',
          'volume-graph-0': '#E5FF0D',
          'volume-graph-1': '#FF820D',
          'ex-negative-sentiment': 'rgb(169, 50, 38)',
          'ex-positive-sentiment': 'rgb(35, 155, 86)',
          'sl-negative-sentiment': 'rgb(241, 148, 138)',
          'sl-positive-sentiment': 'rgb(171, 235, 198)'}

button_style = {'display': 'inline-block',
                    'height': '50px',
                    'padding': '0px 50px',
                    'margin': '0px 20px 20px 20px',
                    'color': colors['table-text'],
                    'text-align': 'center',
                    'font-size': '12px',
                    'font-weight': '600',
                    'line-height': '38px',
                    'letter-spacing': '.1rem',
                    'text-transform': 'uppercase',
                    'text-decoration': 'none',
                    'white-space': 'nowrap',
                    'background-color': 'transparent',
                    'border-radius': '7px',
                    'border': '1px solid #bbb',
                    'cursor': 'pointer',
                    'box-sizing': 'border-box'}
def encode_image(image_file):
    """
    Encodes image files into base64 that can be used/read by Dash/Plotly.
    """
    encoded = base64.b64encode(open(image_file,'rb').read())
    print('Uploading the word cloud')
    return 'data:image/png;base64,{}'.format(encoded.decode())

#### APP ####
app = dash.Dash(__name__, external_stylesheets=['https://www.w3schools.com/w3css/4/w3.css']) # external css style

app.layout = html.Div(className='w3-row', children=[
    ### Main Title ###
    html.Div(className='w3-row', children=[html.H1('Real-Time Product/Service Seniment Analyzer', id='main-title', style={'color':colors['text']}),
                                           html.H3('Author: Sung Bae', id='main-author', style={'color':colors['table-text']}),
                                           html.H4('Last Update: 12.11.2020', id='last-update', style={'color':colors['table-text']})],
                                                    style={'textAlign':'center',
                                                           'padding':'10px'}),
    html.Hr(),

    # ### Keywords for Tweet Filter ###
    # html.Div(className='w3-container', children=[html.Div(className='w3-threequarter', children=[html.Label('Streaming Query (separate by commas)', style={'color':colors['text']}),
    #                                                                                              dcc.Input(id='queries', className='w3-input', type='text', 
    #                                                                                                        placeholder='Ex. Microsoft, Apple, Google', debounce = True, 
    #                                                                                                        style={'backgroundColor':'#202020', 'color': colors['table-text']}),
    #                                                                                              html.Label('Current Keywords: ', style={'color': colors['text']}),
    #                                                                                              html.Div(className='w3-row', id='query-display', style={'color': '#444444'})]),
    #                                             html.Div(className='w3-quarter w3-row-cell', children=[html.Button('Reset Queries', id='reset-queries-button', style=button_style, className='w3-cell-middle')],
    #                                                      style={'textAlign': 'center'})
    #                                             ]),
    html.Hr(),
    ### Live Sentiment Graph Keyword Input ###
    html.Div(className='w3-container', children=[html.Label('Search Brand/Service (Comparison up to two terms separated by comma): ', style={'color':colors['text']}),
                                                 dcc.Input(id='sentiment-term', className='w3-input', type='text', 
                                                           placeholder='Ex. Microsoft, Apple', debounce = True, 
                                                           style={'backgroundColor':'#202020', 'color': colors['table-text'], 'width':'800px'})
                                                ]),

    ### Live Sentiment Graph Control ###
    html.Div(className='w3-container', children=[html.Label('Time Bins: ', style={'color':colors['text']}),
                                                    dcc.RadioItems(id='sentiment-real-time-bin',
                                                                   options=[{'label':i, 'value':i} for i in ['1s', '5s', '10s', '30s', '60s','2min','5min']],
                                                                   value = '1s',
                                                                   style = {'color':colors['text']}),
                                                    html.Label('Maximum Number of Data: ', id='maximum-number-of-data', style={'color':colors['text']}),
                                                    dcc.RadioItems(id='sentiment-max-length',
                                                                   options=[{'label':i, 'value':i} for i in [100, 500, 1000, 2000, 5000, 10000]],
                                                                   value = 500,
                                                                   style = {'color':colors['text']})],
                            
            ),

    ### Live Sentiment Graph ###
    html.Div(className='w3-cell-row', children=[html.Div(dcc.Graph(id='live-sentiment-graph', animate=False), className='w3-container w3-twothird'),
                                           html.Div(dcc.Graph(id='live-tweet-table-pie', animate=False), className='w3-container w3-third w3-cell-middle')]),

    html.Hr(),
    
    ### Generate Word Cloud Upon Clicking the "Generate Button" ###
    html.Div(className='w3-conatiner', children=[html.Button('Generate Word Cloud!', id='get-word-cloud-button',
                                                    style=button_style),
                                        # html.Div(id='word-cloud-loading-message', style={'textAlign':'center', 'color':colors['table-text'], 'padding':'10px'}),
                                        html.Div(className='w3-container', children=[html.Div(html.Img(id='word-cloud-image', src="",
                                                                                                          style={'object-fit': 'cover',
                                                                                                                 'height':'1100px', 
                                                                                                                 'width':'1100px',
                                                                                                                 }
                                                                                                                 ), className='w3-row')])],
                                                    style={'textAlign': 'center'}),
    
                         
    html.Hr(),
    ### Live Tweets Table Title ###
    html.Div(className='w3-cell-row', children=[html.Div(className='w3-container w3-half w3-cell-middle', 
                                                         children=[html.H2(id='live-tweet-table-title', style={'textAlign':'center',
                                                                                                                'color':colors['text'],
                                                                                                                'padding':'13px'})]),
                                                html.Div(className='w3-container w3-half w3-cell-middle', 
                                                         children=[html.H2(id='live-flagged-tweet-table-title', style={'textAlign':'center',
                                                                                                                'color':colors['sl-negative-sentiment'],
                                                                                                                'padding':'13px'})])                   
                                                # html.Div(className='w3-container w3-half w3-center', 
                                                #          children=[html.Button("UPDATE FLAGGED TWEETS!", id='live-flagged-tweet-table-update-button',
                                                #                                 style=flag_button_style)])
                                                ]),
    ### Live Tweets Tables ###
    html.Div(className='w3-cell-row', children=[html.Div(className='w3-container w3-half', 
                                                         children=[html.Div(id="live-tweet-table")]),
                                                                   
                                                html.Div(className='w3-container w3-half w3-center', 
                                                         children=[html.Div(id="live-flagged-tweet-table")])                   
                                                ]),
    html.Hr(),
    ### Data Download Buttons ###
    html.Div(className='w3-container', children=[html.Div(className='w3-container w3-center', 
                                                         children=[html.A('Download Raw Data', id='download-raw-link', 
                                                                                               download="", 
                                                                                               href="", 
                                                                                               target="_blank", 
                                                                                               className='w3-button w3-green'),
                                                                   html.Button('Generate CSV file', id='generate-csv-button', style=button_style),
                                                                   html.A('Download Flagged Data', id='download-flagged-link', 
                                                                                                   download="", 
                                                                                                   href="", 
                                                                                                   target="_blank", 
                                                                                                   className='w3-button w3-red'),
                                                 html.Div(className='w3-row',
                                                         children=[html.Label('Make sure you click "GENERATE CSV FILE" button BEFORE you press download buttons!',
                                                                  style={"color":colors['table-text'], "fontSize":"15px"})]),
                            
                                                                    ])
                                                 ]),
    html.Hr(),

    ### Update Time Interval ###
    dcc.Interval(
        id='live-sentiment-graph-update',
        interval=refresh_time*1000, # in milliseconds
        n_intervals=0
    ),

    dcc.Interval(
        id='live-tweet-table-update',
        interval=(refresh_time+1)*1000, # in milliseconds
        n_intervals=0
    ),

    dcc.Interval(
        id='live-tweet-table-pie-update',
        interval=(refresh_time+0.5)*1000, # in milliseconds
        n_intervals=0
    ),

    dcc.Interval(
        id='live-flagged-tweet-table-update',
        interval=(refresh_time+2)*1000, # in milliseconds
        n_intervals=0
    ),

    # html.Div([
    #     html.Label(['Companies']),
    #     dcc.Dropdown(
    #         id='stock-companies',
    #         options=[
    #                  {'label': 'TSLA', 'value': 'TSLA'},
    #                  {'label': 'AAPL', 'value': 'AAPL'},
    #                  {'label': 'MSFT', 'value': 'MSFT'},
    #                  {'label': 'YHOO', 'value': 'YHOO'},
    #                  {'label': 'GOOGL', 'value': 'GOOGL'},
    #         ],
    #         value = 'Select Companies',
    #         multi=True,
    #         clearable=False,
    #         style={"width": "40%"}),
    #     dcc.RadioItems(id='starting-year',
    #                options=[
    #                    {'label': '2015', 'value': 2015},
    #                    {'label': '2016', 'value': 2016},
    #                    {'label': '2017', 'value': 2017},
    #                    {'label': '2018', 'value': 2018},
    #                    {'label': '2019', 'value': 2019},],
    #                 value = 2015,
    #                 style = {'color':colors['text']}
    #                 )
    # ]),
],
    # Defines overall style
    style = {'backgroundColor': colors['background'], 'margin-top':'10px', 'height':'auto'}
)

############################################
############### FUNCTIONS ##################
############################################

def df_resample(dataframe, time_bin):
    """
    resamples dataframe according to time bin chose by an user.
    """
    vol_df = dataframe.copy()
    vol_df['volume'] = 1
    vol_df = vol_df.resample(time_bin).sum()
    vol_df.dropna(inplace=True)

    dataframe = dataframe.resample(time_bin).mean()
    dataframe.dropna(inplace=True)

    return dataframe.join(vol_df['volume'])


### Word Cloud Related Codes ###
# Code inspired from https://towardsdatascience.com/create-word-cloud-into-any-shape-you-want-using-python-d0b88834bc32
def similar_color_func_blue(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None, color='blue'):
    """
    gives similar colors that will be used in wordcloud
    """
    color_dict = {'blue': 191, 'orange': 30}
    h = color_dict[color] # 0 - 360
    s = 100 # 0 - 100
    l =  np.random.randint(30, 70) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)

def similar_color_func_orange(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None, color='orange'):
    """
    gives similar colors that will be used in wordcloud
    """
    color_dict = {'blue': 191, 'orange': 30}
    h = color_dict[color] # 0 - 360
    s = 100 # 0 - 100
    l =  np.random.randint(30, 70) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)

def get_word_cloud(df_list, keywords):
    """
    Generates two word clouds. One for positive sentiments and one for negative sentiments.
    Args:
        dataframe (pd.DataFrame): dataframe that contains tweet texts and sentiments.

        term (str): key term used to generte the dataframe.
    Returns:
        an image file with two word clouds
    """
    mask = np.array(Image.open('images/tweet_mask.jpg'))
    # print('generating word cloud', term)
    stopword_list = list(STOP_WORDS) + list(string.punctuation) + keywords + ['like', 'good', 'bad', 'fuck', 'fucking', 'hate', 'best', 'awesome', 'great', 'horrible']

    wc_pos = WordCloud(mask=mask, background_color=colors['background'], stopwords=stopword_list,
                        max_font_size=256,
                        random_state=42, width=mask.shape[1]*1.2,
                        height=mask.shape[0]*1.2, color_func=similar_color_func_blue)
    

    wc_neg = WordCloud(mask=mask, background_color=colors['background'], stopwords=stopword_list,
                        max_font_size=256,
                        random_state=42, width=mask.shape[1]*1.2,
                        height=mask.shape[0]*1.2, color_func=similar_color_func_orange)
    
    fig, axes = plt.subplots(nrows=len(keywords), ncols=2, figsize=(24,14*(len(keywords))))

    axes = axes.ravel()
    for i,j in zip(range(len(keywords)), range(0, len(keywords)*2,2)) :
        wc_pos.generate(','.join(df_list[i][df_list[i]['sentiment']>POS_THRESH]['clean']))
        wc_neg.generate(','.join(df_list[i][df_list[i]['sentiment']<NEG_THRESH]['clean']))

        print(f"j: {j}")
        axes[j].axis('off')
        axes[j].imshow(wc_pos, interpolation="bilinear")
        axes[j].set_title(f'Positive Sentiment - {keywords[i].capitalize()}', fontdict={'fontsize': 25, 'fontweight': 'medium', 'color': 'white'})

        axes[j+1].axis('off')
        axes[j+1].imshow(wc_neg, interpolation="bilinear")
        axes[j+1].set_title(f'Negative Sentiment - {keywords[i].capitalize()}', fontdict={'fontsize': 25, 'fontweight': 'medium', 'color': 'white'})
    
    # timestamp = str(datetime.datetime.now())
    fig.subplots_adjust(hspace=-0.5, wspace=.001, top=0.99, bottom=0.01)
    # plt.tight_layout()
    img_file = 'images/word_cloud.png'
    fig.savefig(img_file, facecolor=colors['background'], edgecolor=colors['background'])
    # print('creted word cloud files and about to save it!')
    plt.close('all')
    return img_file



def generate_tweet_table(dataframe):
    """
    Generates tweets table for dash
    """
    return dash_table.DataTable(id="responsive-table",
                                columns=[{'name': 'Date', 'id':'date', 'type': 'datetime'},
                                         {'name': 'Tweet', 'id':'tweet', 'type': 'text'},
                                         {'name': 'Sentiment', 'id':'sentiment', 'type': 'numeric'},
                                         {'name': 'Link', 'id':'link', 'type': 'text', 'presentation':'markdown'}],
                                data = dataframe.to_dict('records'),
                                style_header={
                                    'backgroundColor': 'rgb(52, 73, 94)',
                                    'fontWeight': 'bold',
                                    'color': colors['text'],
                                    'textAlign': 'left',
                                    'fontSize': '12pt',
                                    'height': 'auto',
                                    'width': 'auto'
                                },
                                style_cell={'padding': '5px',
                                            'backgroundColor': colors['background'],
                                            'color': colors['table-text'],
                                            'textAlign':'left',
                                            'height':'auto',
                                            'whiteSpace':'normal',
                                            'lineHeight':'15px',
                                            'width':'auto'},
                                style_as_list_view=True,
                                style_data_conditional=[
                                    {
                                        'if': {
                                            'filter_query': '{sentiment} < -0.3'
                                        },
                                        'backgroundColor': colors['sl-negative-sentiment'],
                                        'color': colors['ex-negative-sentiment']
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{sentiment} < -0.6'
                                        },
                                        'backgroundColor': colors['ex-negative-sentiment'],
                                        'color': 'white'
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{sentiment} > 0.3'
                                        },
                                        'backgroundColor': colors['sl-positive-sentiment'],
                                        'color': colors['ex-positive-sentiment']
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{sentiment} > 0.6'
                                        },
                                        'backgroundColor': colors['ex-positive-sentiment'],
                                        'color': 'white'
                                    },
                                ]),

def generate_flagged_tweet_table(dataframe):
    """
    Generates table for flatted tweets that will be displayed in Dash.
    """
    return dash_table.DataTable(id="flagged-table",
                                columns=[{'name': 'Date', 'id':'date', 'type': 'datetime'},
                                         {'name': 'Tweet', 'id':'tweet', 'type': 'text'},
                                         {'name': 'Sentiment', 'id':'sentiment', 'type': 'numeric'},
                                         {'name': 'Link', 'id':'link', 'type': 'text', 'presentation':'markdown'},
                                         {'name': 'Dealt', 'id':'dealt', 'type': 'text', 'presentation':'dropdown'},],
                                data = dataframe.to_dict('records'),
                                editable = True,
                                dropdown={'dealt': {'options': [{'label': str(i), 'value': i} for i in [0, 1]]}},
                                style_header={
                                    'backgroundColor': 'rgb(52, 73, 94)',
                                    'fontWeight': 'bold',
                                    'color': colors['text'],
                                    'textAlign': 'left',
                                    'fontSize': '12pt',
                                    'height': 'auto',
                                    'width': 'auto'
                                },
                                style_cell={'padding': '5px',
                                            'backgroundColor': colors['background'],
                                            'color': colors['table-text'],
                                            'textAlign':'left',
                                            'height':'auto',
                                            'whiteSpace':'normal',
                                            'lineHeight':'15px',
                                            'width':'auto'},
                                style_as_list_view=True,
                                style_data_conditional=[
                                    {
                                        'if': {
                                            'filter_query': '{sentiment} < -0.3'
                                        },
                                        'backgroundColor': colors['sl-negative-sentiment'],
                                        'color': colors['ex-negative-sentiment']
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{sentiment} < -0.6'
                                        },
                                        'backgroundColor': colors['ex-negative-sentiment'],
                                        'color': 'white'
                                    }
                                ]),
def make_clickable(id):
    """
    Generates a link that can be used in Dash table.
    Args:
        id (str): Tweet id in string format
    Returns:
        link (str): Lnk that can be used in dash table.
    """
    return f'[Link](https://twitter.com/user/status/{id})'


def get_df(keywords, max_length, combine=False):
    df = []
    if combine and len(keywords)>1:
        q = ' OR '.join([f"tweet LIKE '%{word}%'" for word in keywords])
        q = f'({q})'
        df.append(pd.read_sql(f"SELECT * FROM sentiment WHERE {q} ORDER BY unix DESC LIMIT {max_length}", conn))
    else:
        if len(keywords): 
            for i, word in enumerate(keywords):
                if i >= 2:
                    break
                df.append(pd.read_sql(f"SELECT * FROM sentiment WHERE tweet LIKE '%{word}%' ORDER BY unix DESC LIMIT {max_length}", conn))
        else: # if no keyword was given
            df.append(pd.read_sql(f"SELECT * FROM sentiment ORDER BY unix DESC LIMIT {max_length}", conn))#, params=('%' + term + '%'))    

    return df
#########################################################
##################### CALL BACKS ########################
#########################################################

# ### QUERY SELECTION ###
# @app.callback(Output('query-display', 'children'),
#               [Input('queries', 'value'),
#                Input('reset-queries-button', 'n_clicks')])
# def update_query(keywords, n_clicks):
#     print('triggered')
#     new = False
#     ctx = dash.callback_context
#     global CURRENT_KEYWORDS
#     if 'queries' == ctx.triggered[0]['prop_id'].split('.')[0]:
#         # print(ctx.triggered[0]['value'])
#         word_list = ctx.triggered[0]['value'].split(',')
#         for word in word_list:
#             if word.lower() not in CURRENT_KEYWORDS and len(word)>0:
#                 print(f'new word:{word}')
#                 CURRENT_KEYWORDS.append(word.lower())
#                 new = True
#         if new:
#             streamer.update_stream(CURRENT_KEYWORDS)
#         return ', '.join(CURRENT_KEYWORDS)
#     elif 'n_clicks' == ctx.triggered[0]['prop_id'].split('.')[0]:
#         if n_clicks:
#             print('n_clicks: ',n_clicks)
#             CURRENT_KEYWORDS = []
#             streamer.twitterStream.disconnect()
#             return CURRENT_KEYWORDS

# ### QUERY RESET ###
# @app.callback(Output('query-display', 'children'),
#               [Input('reset-queries-button', 'n_clicks')])
# def reset_query(n_clicks):
#     CURRENT_KEYWORDS = []
#     return CURRENT_KEYWORDS

### LIVE SENTIMENT GRAPH ###
@app.callback(Output('live-sentiment-graph', 'figure'),
              [Input('live-sentiment-graph-update', 'n_intervals'),
               Input('sentiment-term', 'value'),
               Input('sentiment-real-time-bin', 'value'),
               Input('sentiment-max-length', 'value')])
def update_sentiment_graph(n, term, time_bin, max_length):
    # Adjusts refresh time when the number of tweets increases to reduce stress on the system.
    global refresh_time
    if max_length == 1000:
        refresh_time = 5
    elif max_length == 2000:
        refresh_time = 10
    elif max_length >= 5000:
        refresh_time = 15
    # print('starting over: reinitializing everything...')
    df_sentiment = []
    resampled_df = []
    fig = go.Figure()
    Y = []
    Y2 = []
    X = []
    df = []

    try:
        # Generates a global dataframe that can be used by other callbacks once pulled from SQLite3 database
        if term: 
            keywords = [x.strip() for x in term.split(',')]
            title = f'Live Sentiment Graph - {[x.capitalize() for x in keywords]}'
            
        else: # if no keyword was given
            keywords = []
            title = 'Live Sentiment Graph'
            
        df = get_df(keywords, max_length)
        for i in range(len(df)):
            # max_num = df[i].shape[0] if max_length > df[i].shape[0] else max_length
            # Makes a copy of df to keep the original
            df_sentiment.append(df[i].copy())

            # Converts unix into datetime and set it as index
            df_sentiment[i].sort_values('unix', inplace=True)
            df_sentiment[i]['date'] = pd.to_datetime(df_sentiment[i]['unix'], unit='ms')
            df_sentiment[i].set_index('date', inplace=True)
            
            # Rolling average to smooth out the graph
            df_sentiment[i]['sentiment_smoothed'] = df_sentiment[i]['sentiment'].rolling(int(len(df[i])/10)).mean()

            # Resamples according to time_bin value set by an user
            # time_period = f'{int((df_sent.index[i]-df_sent.index[-1]).seconds/time_bin)}s'

            resampled_df.append(df_resample(df_sentiment[i], time_bin))
 
            Y.append(list(resampled_df[i].sentiment_smoothed.values))
            Y2.append(list(resampled_df[i].volume.values))
            X.append(list(resampled_df[i].index))
            # Creates Plotly graph for real-time sentiment
            if len(keywords):
                name_1 = f'Scatter - {keywords[i]}'
                name_2 = f'Volume - {keywords[i]}'
            else:
                name_1 = 'Scatter'
                name_2 = 'Volume'
            fig.add_trace(go.Scatter(x = X[i],
                                     y = Y[i],
                                     name = name_1,
                                     yaxis='y2',
                                     mode = 'lines+markers',
                                     line = {"color": colors[f'sentiment-graph-{i}'], "width":2}),
                            )

            fig.add_trace(go.Bar(x = X[i],
                                 y = Y2[i],
                                 name = name_2,
                                 opacity = 1,
                                 marker = dict(color = (colors[f'volume-graph-{i}']))
                                ))
        
        X_total = [x for x_list in X for x in x_list]   
        Y_total = [x for x_list in Y for x in x_list]   
        Y2_total = [x for x_list in Y2 for x in x_list]     
        print(len(X_total), len(Y_total), len(Y2_total))          
        fig.update_layout(xaxis = dict(range=[min(X_total), max(X_total)], linecolor='black'),
                            yaxis = dict(range=[0, max(Y2_total)*3], title='Volume', side='right', showgrid=False),
                            yaxis2 = dict(range=[min(Y_total)*1.2 if min(Y_total)<0 else -max(Y_total)*0.3, 
                                                max(Y_total)*1.2 if max(Y_total)>0 else 0.1], title='Sentiment', overlaying='y', side='left', showgrid=False), 
                            title = title, title_x=0.5,
                            font = dict(color=colors['text']),
                            plot_bgcolor = colors['background'],
                            paper_bgcolor = colors['background'])


        return fig

    except Exception as e:
        print(e)


## LIVE WORD CLOUD UPDATE ###
@app.callback(Output('word-cloud-image', 'src'),
              [Input('get-word-cloud-button', 'n_clicks')],
              [State('sentiment-term', 'value'),
               State('sentiment-max-length', 'value')])
def update_word_cloud(n_clicks, term, max_length):
    """
    Generates word cloud.
    """
    if n_clicks:
        print(n_clicks)
        if term: 
            keywords = [x.strip() for x in term.split(',')]
            df = get_df(keywords, max_length)
            img_file = get_word_cloud(df, keywords)
        else: # if no keyword was given
            df = get_df([], max_length)
            img_file = get_word_cloud(df, keywords=[''])
        return encode_image(img_file)
    else:
        return encode_image('images/cloud.jpeg') # https://unsplash.com/photos/H83_BXx3ChY by @dallasreedy



# LIVE TWEEET TABLE UPDATE ###
@app.callback(Output('live-tweet-table', 'children'),
              Output('live-tweet-table-title', 'children'),
              [Input('live-tweet-table-update', 'n_intervals'),
               Input('sentiment-term', 'value')])        
def update_tweet_table(n_interval, term):
    # print(f"live tweet table n_interval: {n_interval}")
    if term: 
        keywords = [x.strip() for x in term.split(',')]
        
    else: # if no keyword was given
        keywords = []
    
    df = get_df(keywords, 20, combine=True)[0]

    MAX_ROWS=15
    df_table = df.copy()
    df_table['date'] = pd.to_datetime(df_table['unix'], unit='ms')
    df_table['link'] = df_table['id'].apply(make_clickable)
    df_table = df_table[['date','tweet','sentiment', 'link']].iloc[:MAX_ROWS]
    
    if len(keywords):
        title = f'Recent Tweets - {keywords}'
    else:
        title = 'Recent Tweets'

    return generate_tweet_table(df_table), title

def make_pie_chart(df_pie, max_length):
    
    max_length = df_pie.shape[0] if max_length > df_pie.shape[0] else max_length
    labels = ['Positive', 'Negative', 'Neutral']
    pos = sum(df_pie['sentiment'] > 0)
    neg = sum(df_pie['sentiment'] < 0)
    neu = sum(df_pie['sentiment'] == 0)
    colors_pie = ['#58D68D', '#E74C3C', '#F7DC6F']

    return go.Pie(title=f'(n={max_length})',
                  labels=labels, values = [pos, neg, neu],
                    hoverinfo='label+percent', textinfo='value',
                    textfont=dict(size=20, color=colors['background']), #'#566573'),
                    marker=dict(colors=colors_pie,
                                line=dict(color=colors['background'], width=1)))

## LIVE PIE UPDATE ###
@app.callback(Output('live-tweet-table-pie', 'figure'),
              [Input('live-tweet-table-pie-update', 'n_intervals'),
               Input('sentiment-term', 'value'),
               Input('sentiment-max-length', 'value')])        
def update_pie(n_interval, term, max_length):
    # print(f"Live-tweet-pie n_interval: {n_interval}")
    # if n_interval<2:
    #     time.sleep(refresh_time)
    
    if term:
        keywords = [x.strip() for x in term.split(',')]
    else:
        keywords = []

    if len(keywords) >= 2:
        df = get_df(keywords[:2], max_length)
        fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                            subplot_titles=(f'{keywords[0].capitalize()}', f'{keywords[1].capitalize()}'))
        print('length of df: ', len(df))
        for i in range(len(df)):
            pie = make_pie_chart(df[i], max_length)
            fig.add_trace(pie, row=1, col=(i+1))

        title = f'Sentiment Distribution Charts'
        fig.update_layout(title_text=title, title_x=0.5,
                            font={'color': colors['text']},
                            plot_bgcolor = colors['background'],
                            paper_bgcolor = colors['background'],
                            showlegend = True)

    else: # if no keyword was given
        df = get_df(keywords, max_length)
        # print('im here', len(df), df[0].shape)
        fig = go.Figure()
        # print(f'hello, {len(df)}, {df[0].shape}')
        pie = make_pie_chart(df[0], max_length)
        fig.add_trace(pie)

        if len(keywords) == 1:
            title= f'Sentiment Distribution Charts - {keywords}'
        else:
            title = 'Sentiment Distribution Charts'

        print(f'type: {type(fig)}, keywords: {keywords}')
        fig.update_layout(title=title,
                            font={'color': colors['text']},
                            plot_bgcolor = colors['background'],
                            paper_bgcolor = colors['background'],
                            showlegend = True)
    # print('uhhhh?')
    return fig

# LIVE FLAGGED TWEEET TABLE UPDATE ###
@app.callback(Output('live-flagged-tweet-table', 'children'),
              Output('live-flagged-tweet-table-title', 'children'),
              [Input('live-flagged-tweet-table-update', 'n_intervals'),
               Input('sentiment-term', 'value')])    
def update_flagged_tweet_table(n_interval, term):
    # print(f"Flag n_interval: {n_interval}")
    if n_interval<2:
        time.sleep(refresh_time*2)
    if term:
        keywords = [x.strip() for x in term.split(',')]
    else:
        keywords = []

    if len(keywords)==2:
        q = ' OR '.join([f"tweet LIKE '%{word}%'" for word in keywords])
        q = f'({q})'
        flagged_df = pd.read_sql(f"SELECT * FROM flag WHERE {q} ORDER BY unix DESC LIMIT 30", conn)
    elif len(keywords)==1:
        flagged_df = pd.read_sql("""SELECT * FROM flag 
                                    WHERE tweet LIKE '%{}%' AND dealt != 1
                                    ORDER BY unix DESC LIMIT 30""".format(keywords[0]), conn)
    else:
        flagged_df = pd.read_sql("""SELECT * FROM flag 
                                    WHERE dealt != 1
                                    ORDER BY unix DESC LIMIT 30""", conn)
    MAX_ROWS=15
    flagged_df['date'] = pd.to_datetime(flagged_df['unix'], unit='ms')
    flagged_df['link'] = flagged_df['id'].apply(make_clickable)
    flagged_df = flagged_df[['date','tweet','sentiment', 'link', 'dealt']].iloc[:MAX_ROWS]
    
    if term:
        title = f'Recently Flagged Tweets - {keywords}'
    else:
        title = 'Recent Flagged Tweets'
    return generate_flagged_tweet_table(flagged_df), title

@app.callback(
    Output('download-raw-link', 'href'),
    Output('download-raw-link', 'download'),
    # Output('download-raw-link', 'className'),
    Output('download-flagged-link', 'href'),
    Output('download-flagged-link', 'download'),
    # Output('download-flagged-link', 'className'),
    [Input('generate-csv-button', 'n_clicks')],
    [State('sentiment-term', 'value'),
     State('sentiment-max-length', 'value')])
def update_download_link(n_clicks, term, max_length):
    if term:
        keywords = [x.strip() for x in term.split(',')]
    else:
        keywords = []

    if len(keywords)==2:
        q = ' OR '.join([f"tweet LIKE '%{word}%'" for word in keywords])
        q = f'({q})'

        raw_df = pd.read_sql(f"SELECT * FROM sentiment WHERE {q} ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))
        raw_title = f'live_tweet_data_{keywords[0]}_{keywords[1]}_{raw_df.shape[0]}.csv'
        flagged_df = pd.read_sql(f"SELECT * FROM flag WHERE {q} ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))
        flagged_title = f'live_flagged_tweet_data_{keywords[0]}_{keywords[1]}_{flagged_df.shape[0]}.csv'
    elif len(keywords)==1:
        raw_df = pd.read_sql(f"SELECT * FROM sentiment WHERE tweet LIKE '%{keywords[0]}%' ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))
        raw_title = f'live_tweet_data_{keywords[0]}_{raw_df.shape[0]}.csv'
        flagged_df = pd.read_sql(f"SELECT * FROM flag WHERE tweet LIKE '%{keywords[0]}%' ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))
        flagged_title = f'live_flagged_tweet_data_{keywords[0]}_{flagged_df.shape[0]}.csv'

    else: # if no keyword was given
        raw_df = pd.read_sql(f"SELECT * FROM sentiment ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))    
        raw_title = f'live_tweet_data_{raw_df.shape[0]}.csv'
        flagged_df = pd.read_sql(f"SELECT * FROM flag ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))    
        flagged_title = f'live_flagged_tweet_data_{flagged_df.shape[0]}.csv'
    
    raw_csv_string = raw_df.to_csv(index=False, encoding='utf-8')
    raw_csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(raw_csv_string)

    flagged_csv_string = flagged_df.to_csv(index=False, encoding='utf-8')
    flagged_csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(flagged_csv_string)

    return raw_csv_string, raw_title, flagged_csv_string, flagged_title


################################################################
# @app.callback(
#     Output(component_id='output-graph', component_property='children'),
#     [Input(component_id='stock-companies', component_property='value'),
#      Input('starting-year', 'value')]
# )
# def update_graph(company_names, starting_year):
#     start = datetime.datetime(starting_year, 1, 1)
#     end = datetime.datetime.now()
#     data = []
#     for company in company_names:
#         df = web.DataReader(company, 'yahoo', start, end)
#         data.append({'x':df.index, 'y':df.Close, 'type': 'line', 'name': company})

#     return dcc.Graph(id='example-stock',
#               figure = {
#                 'data': data,
#                 'layout': {
#                     'title': 'Stock Graphs!',
#                     'plot_bgcolor':colors['background'],
#                     'paper_bgcolor':colors['background'], 
#                     'font': {'color':colors['text']},
#                 }
#               }
#               )
# @app.callback(
#     Output('print-company-names', 'children'),
#     [Input('stock-companies', 'value')]
# )
# def print_company_names(value):
#     return "Options: {}".format(value)

# @app.callback(
#     Output(component_id='output_num', component_property='children'),
#     [Input(component_id='input_num', component_property='value')]
# )
# def update_value(input_data):
#     try:
#         return str(float(input_data)**2)
#     except:
#         return "Some error"
#     return "Input: {}".format(input_data)
##########################################################
############### END OF INTER COMPONENTS ##################
##########################################################

#### Materializing CSS ####
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
# external_css = ["https://www.w3schools.com/w3css/4/w3.css"]
# for css in external_css:
#     app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=True)