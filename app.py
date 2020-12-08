import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import sqlite3 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go
from collections import deque
import plotly.express as px
import random

import pandas_datareader.data as web
import datetime

import string
import re
import spacy
from PIL import Image
nlp = spacy.load('en_core_web_lg')
from wordcloud import WordCloud
import numpy as np

import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS
import ktrain

### SQL Connection ###
conn = sqlite3.connect('data/twitter_2.db', check_same_thread=False)

### Bert Model ####
MODEL_PATH = 'models/BERT_2'
predictor = ktrain.load_predictor(MODEL_PATH)

#### STYLE ####
colors = {'background': '#111111', 
          'text':'#7FDBFF',
          'table-text':'CDD1CA',
          'sentiment-graph': '#00FFE8',
          'volume-graph': '#FFFF00',
          'ex-negative-sentiment': 'FF2D00',
          'ex-positive-sentiment': '00EC0E',
          'sl-negative-sentiment': 'FF8300',
          'sl-positive-sentiment': '81DC3A'}

#### APP ####
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.Div(className='container-fluid', children=[html.H1('Dashboard - Tweet Analysis')],
                                                    style={'textAlign':'center',
                                                           'color':colors['text'],
                                                           'padding':'10px'}),
    html.Hr(),
    ### Real-time Data ###
    html.Div(className='container-fluid', children=[html.Label('Search: ', style={'color':colors['text']}),
                                                    dcc.Input(id='sentiment-term', value='Microsoft', type='text')],
                                                    style={'textAlign':'left',
                                                           'padding':'5px',
                                                           'color':colors['text']}),

    html.Div(className='container-fluid', children=[html.Label('Time Bins: ', style={'color':colors['text']}),
                                                    dcc.RadioItems(id='sentiment-real-time-bin',
                                                                   options=[{'label':i, 'value':i} for i in ['1s', '5s', '10s']],
                                                                   value = '1s',
                                                                   style = {'color':colors['text']}),
                                                    html.Label('Maximum Number of Data: ', style={'color':colors['text']}),
                                                    dcc.RadioItems(id='sentiment-max-length',
                                                                   options=[{'label':i, 'value':i} for i in [100, 500, 1000]],
                                                                   value = 100,
                                                                   style = {'color':colors['text']})],
                                                    style={'textAlign':'left',
                                                           'padding':'5px',
                                                           'color':colors['text']}),
    html.Hr(),
    # Live Graph Test
    html.Div(className='row', children=[html.Div(dcc.Graph(id='live-sentiment-graph', animate=False), className='col s12 m6 l6'),
                                        html.Div(html.Img(src='testplot.png'), className='col s12 m6 l6')]),

    html.Div(className='row', children=[html.Div(id="live-tweet-table", className='col s12 m6 l6'),
                                        html.Div(html.Label('Place Holder', style={'color':colors['text']}), className='col s12 m6 l6'),]),
    dcc.Interval(
        id='live-sentiment-graph-update',
        interval=1*1000, # in milliseconds
        n_intervals=0
    ),

    # dcc.Interval(
    #     id='live-word-cloud-update',
    #     interval=60*1000, # in milliseconds
    #     n_intervals=0
    # ),

    dcc.Interval(
        id='live-tweet-table-update',
        interval=2*1000, # in milliseconds
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


    dcc.Input(id='input', value='', type='text'),
    html.Div(id='output-graph'),

    html.Div([
        html.Label('Input Field'),
        dcc.Input(id='input_num', value='Enter something', type='text'),
        html.Div(id='output_num'),
    ]),
    
],
    # Defines overall style
    style = {'backgroundColor': colors['background'], 'margin-top':'10px', 'height':'2000px'}
)
############################################
############### FUNCTIONS ##################
############################################

def df_resample(df, time_bin):
    vol_df = df.copy()
    vol_df['volume'] = 1
    vol_df = vol_df.resample(time_bin).sum()
    vol_df.dropna(inplace=True)

    df = df.resample(time_bin).mean()
    df.dropna(inplace=True)

    return df.join(vol_df['volume'])


### Word Cloud Related Codes ###
# Code inspired from https://towardsdatascience.com/create-word-cloud-into-any-shape-you-want-using-python-d0b88834bc32
def similar_color_func_blue(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None, color='blue'):
    color_dict = {'blue': 191, 'orange': 30}
    h = color_dict[color] # 0 - 360
    s = 100 # 0 - 100
    l =  np.random.randint(30, 70) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)

def similar_color_func_orange(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None, color='orange'):
    color_dict = {'blue': 191, 'orange': 30}
    h = color_dict[color] # 0 - 360
    s = 100 # 0 - 100
    l =  np.random.randint(30, 70) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)

# def get_word_cloud(df, term):
#     df['clean'] = df['tweet'].apply(clean_text)

#     mask = np.array(Image.open('images/tweet_mask.jpg'))

#     term = [term]

#     stopword_list = list(STOP_WORDS) + list(string.punctuation) + term

#     wc_pos = WordCloud(mask=mask, background_color="white", stopwords=stopword_list,
#                         max_font_size=256,
#                         random_state=42, width=mask.shape[1]*1.8,
#                         height=mask.shape[0]*1.8, color_func=similar_color_func_blue)
#     wc_pos.generate(','.join(df[df['sentiment']>0]['clean']))

#     wc_neg = WordCloud(mask=mask, background_color="white", stopwords=stopword_list,
#                 max_font_size=256,
#                 random_state=42, width=mask.shape[1]*1.8,
#                 height=mask.shape[0]*1.8, color_func=similar_color_func_orange)
#     wc_neg.generate(','.join(df[df['sentiment']<0]['clean']))

#     fig, axes = plt.subplots(ncols=2, figsize=(30,15))
#     axes[0].axis('off')
#     axes[0].imshow(wc_pos, interpolation="bilinear")
#     axes[0].set_title('Positive Sentiment', fontdict={'fontsize': 50, 'fontweight': 'medium'})

#     axes[1].axis('off')
#     axes[1].imshow(wc_neg, interpolation="bilinear")
#     axes[1].set_title('Negative Sentiment', fontdict={'fontsize': 50, 'fontweight': 'medium'})

#     return fig

def tweet_color(sentiment):
    if sentiment <= -0.5:
        return colors['ex-negative-sentiment']
    elif sentiment < 0:
        return colors['sl-negative-sentiment']
    elif sentiment > 0.5:
        return colors['ex-positive-sentiment']
    elif sentiment > 0.5:
        return colors['sl-positive-sentiment']
    else:
        return colors['background']

def generate_tweet_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[html.Thead(html.Tr(children=[html.Th(col.title()) for col in df.columns.values],
                                                   style={'color':colors['text']})),
                                html.Tbody([html.Tr(children=[html.Td(data) for data in d], 
                                                    style={'color':colors['table-text'], 
                                                           'background-color': tweet_color(d[-1])})
                                                    for d in df.values.tolist()])
                        ])
###################################################
############### END OF FUNCTIONS ##################
###################################################


#########################################################
############### INTERACTIVE COMPONENTS ##################
#########################################################
### LIVE SENTIMENT GRAPH ###
@app.callback(Output('live-sentiment-graph', 'figure'),
              [Input('live-sentiment-graph-update', 'n_intervals'),
               Input('sentiment-term', 'value'),
               Input('sentiment-real-time-bin', 'value'),
               Input('sentiment-max-length', 'value')])
def update_sentiment_graph(n, term, time_bin, max_length):
    df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%{}%' ORDER BY unix DESC LIMIT 5000".format(term), conn)#, params=('%' + term + '%'))
    df.sort_values('unix', inplace=True)
    df['date'] = pd.to_datetime(df['unix'], unit='ms')
    df.set_index('date', inplace=True)
    df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/10)).mean()

    df = df_resample(df, time_bin)
    # df.dropna(inplace=True)

    X = df.index[-max_length:]
    Y = df.sentiment_smoothed.values[-max_length:]
    Y2 = df.volume.values[-max_length:]
    
    data1 = go.Scatter(
        x = list(X),
        y = list(Y),
        name = 'Scatter',
        mode = 'lines+markers',
        yaxis = 'y2',
        line = dict(color = (colors['sentiment-graph']),
                    width = 2)
    )
    data2 = go.Bar(
        x = list(X),
        y = list(Y2),
        name = 'Volume',
        marker = dict(color = (colors['volume-graph']))
    )

    return {'data': [data1, data2],
            'layout': go.Layout(xaxis = dict(range=[min(X), max(X)]),
                                yaxis = dict(range=[0, max(Y2)*3], title='Volume', side='right'),
                                yaxis2 = dict(range=[min(Y)*1.2 if min(Y)<0 else -max(Y)*0.3, 
                                                     max(Y)*1.2 if max(Y)>0 else 0.1], title='Sentiment', overlaying='y', side='left'), 
                                title = 'Sentiment Moving Average - {}'.format(term),
                                font = dict(color=colors['text']),
                                plot_bgcolor = colors['background'],
                                paper_bgcolor = colors['background'])}

### LIVE WORD CLOUD UPDATE ###
# @app.callback(Output('live-word-cloud', 'figure'),
#               [Input('live-word-cloud-update', 'n_intervals'),
#                Input('sentiment-term', 'value')])
# def update_word_cloud(n, term):

#     df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%{}%' ORDER BY unix DESC LIMIT 5000".format(term), conn)#, params=('%' + term + '%'))
#     fig = get_word_cloud(df, term)

#     return fig


### LIVE TWEEET TABLE UPDATE ###
@app.callback(Output('live-tweet-table', 'children'),
              [Input('live-tweet-table-update', 'n_intervals'),
               Input('sentiment-term', 'value')])        
def update_tweet_table(n, sentiment_term):
    if sentiment_term:
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%{}%' ORDER BY unix DESC LIMIT 20".format(sentiment_term), conn)
    else:
        pass

    df['date'] = pd.to_datetime(df['unix'], unit='ms')

    df = df.drop(['unix','id'], axis=1)
    df = df[['date','tweet','sentiment']]

    return generate_tweet_table(df, max_rows=20)

################################################################
@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='stock-companies', component_property='value'),
     Input('starting-year', 'value')]
)
def update_graph(company_names, starting_year):
    start = datetime.datetime(starting_year, 1, 1)
    end = datetime.datetime.now()
    data = []
    for company in company_names:
        df = web.DataReader(company, 'yahoo', start, end)
        data.append({'x':df.index, 'y':df.Close, 'type': 'line', 'name': company})

    return dcc.Graph(id='example-stock',
              figure = {
                'data': data,
                'layout': {
                    'title': 'Stock Graphs!',
                    'plot_bgcolor':colors['background'],
                    'paper_bgcolor':colors['background'], 
                    'font': {'color':colors['text']},
                }
              }
              )
@app.callback(
    Output('print-company-names', 'children'),
    [Input('stock-companies', 'value')]
)
def print_company_names(value):
    return "Options: {}".format(value)

@app.callback(
    Output(component_id='output_num', component_property='children'),
    [Input(component_id='input_num', component_property='value')]
)
def update_value(input_data):
    try:
        return str(float(input_data)**2)
    except:
        return "Some error"
    return "Input: {}".format(input_data)
##########################################################
############### END OF INTER COMPONENTS ##################
##########################################################

#### Materializing CSS ####
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=True)