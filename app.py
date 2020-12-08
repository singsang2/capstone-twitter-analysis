import warnings
warnings.filterwarnings(action='ignore')
import dash_table
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

import base64
from IPython.display import HTML
import time
import datetime

# import ktrain

### SQL Connection ###
conn = sqlite3.connect('data/twitter_2.db', check_same_thread=False, timeout=25)
c = conn.cursor()

### Bert Model ####
# MODEL_PATH = 'models/BERT_2'
# predictor = ktrain.load_predictor(MODEL_PATH)

POS_THRESH = 0.3
NEG_THRESH = -0.3

#### STYLE ####
colors = {'background': '#111111', 
          'text':'#7FDBFF',
          'table-text':'rgb(229, 231, 233)',
          'sentiment-graph': '#00FFE8',
          'volume-graph': '#FFFF00',
          'ex-negative-sentiment': 'rgb(169, 50, 38)',
          'ex-positive-sentiment': 'rgb(35, 155, 86)',
          'sl-negative-sentiment': 'rgb(241, 148, 138)',
          'sl-positive-sentiment': 'rgb(171, 235, 198)'}

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
                                                    dcc.Input(id='sentiment-term', value='Microsoft', type='text', debounce = True)],
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

    # Live Graph Test
    html.Div(className='row', children=[html.Div(dcc.Graph(id='live-sentiment-graph', animate=False), className='col s12 m6 l6')]),
    html.Div(className='row', children=[html.Button('Generate Word Cloud!', id='get-word-cloud-button',
                                                    style={'display': 'inline-block',
                                                            'height': '50px',
                                                            'padding': '0 30px',
                                                            'color': colors['table-text'], #'#555',
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
                                                            'box-sizing': 'border-box'})],
                                                    style={'align': 'center'}),
    html.Div(className='row', children=[html.Div(html.Img(id='word-cloud-image', src='children'), className='col s12 m6 l6')]),                                    
                         
    html.Hr(),
    html.Div(className='container-fluid', children=[html.H2(id='live-tweet-table-title', 
                                                            style={'textAlign':'center',
                                                                   'color':colors['text'],
                                                                   'padding':'10px'})]),
    # LIVE TWEET TABLE
    html.Div(className='row', children=[html.Div(id="live-tweet-table", className='col s12 m6 l6', style={'color':colors['text'], 'background-color':colors['ex-negative-sentiment']}),
                                        html.Div(html.Label('Place Holder', style={'color':colors['table-text']}), className='col s12 m6 l6'),]),
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


    # dcc.Input(id='input', value='', type='text'),
    # html.Div(id='output-graph'),

    # html.Div([
    #     html.Label('Input Field'),
    #     dcc.Input(id='input_num', value='Enter something', type='text'),
    #     html.Div(id='output_num'),
    # ]),
    
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

def get_word_cloud(df, term):
    mask = np.array(Image.open('images/tweet_mask.jpg'))

    term = [term]
    stopword_list = list(STOP_WORDS) + list(string.punctuation) + term

    wc_pos = WordCloud(mask=mask, background_color=colors['background'], stopwords=stopword_list,
                        max_font_size=256,
                        random_state=42, width=mask.shape[1]*1.8,
                        height=mask.shape[0]*1.8, color_func=similar_color_func_blue)
    wc_pos.generate(','.join(df[df['sentiment']>POS_THRESH]['clean']))

    wc_neg = WordCloud(mask=mask, background_color=colors['background'], stopwords=stopword_list,
                        max_font_size=256,
                        random_state=42, width=mask.shape[1]*1.8,
                        height=mask.shape[0]*1.8, color_func=similar_color_func_orange)
    wc_neg.generate(','.join(df[df['sentiment']<NEG_THRESH]['clean']))

    fig, axes = plt.subplots(ncols=2, figsize=(30,15))
    axes[0].axis('off')
    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].set_title('Positive Sentiment', fontdict={'fontsize': 50, 'fontweight': 'medium'})

    axes[1].axis('off')
    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].set_title('Negative Sentiment', fontdict={'fontsize': 50, 'fontweight': 'medium'})
    
    # timestamp = str(datetime.datetime.now())
    img_file = 'images/word_cloud.png'
    fig.savefig(img_file, facecolor=colors['background'], edgecolor=colors['background'])

    return img_file

def encode_image(image_file):
    encoded = base64.b64encode(open(image_file,'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())

def generate_tweet_table(df):
    return dash_table.DataTable(id="responsive-table",
                                columns=[{'name': 'Date', 'id':'date', 'type': 'datetime'},
                                         {'name': 'Tweet', 'id':'tweet', 'type': 'text'},
                                         {'name': 'Sentiment', 'id':'sentiment', 'type': 'numeric'},
                                         {'name': 'Link', 'id':'link', 'type': 'text', 'presentation':'markdown'},
                                         {'name': 'Dealt', 'id':'dealt', 'type': 'text', 'editable':True},],
                                data = df.to_dict('records'),
                                style_header={
                                    'backgroundColor': 'rgb(52, 73, 94)',
                                    'fontWeight': 'bold',
                                    'fontColor': colors['text'],
                                    'textAlign': 'center',
                                    'fontSize': '12pt',
                                    'height': 'auto'
                                },
                                style_cell={'padding': '5px',
                                            'backgroundColor': colors['background'],
                                            'fontColor': colors['table-text'],
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
    # return html.Table(className="responsive-table",
    #                   children=[html.Thead(html.Tr(children=[html.Th(col.title()) for col in df.columns.values],
    #                                                style={'color':colors['text'],
    #                                                       'background-color': tweet_color(-0.7)})),
    #                             html.Tbody([html.Tr(children=[html.Td(data) for data in d], 
    #                                                 style={'color':colors['table-text'], 
    #                                                        'background-color': tweet_color(-0.7)})
    #                                                 for d in df.values.tolist()])
    #                     ])
def make_clickable(id):
    return f'[Link](https://twitter.com/user/status/{id})'

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
    try:
        global df
        if term:
            df = pd.read_sql(f"SELECT * FROM sentiment WHERE tweet LIKE '%{term}%' ORDER BY unix DESC LIMIT 1000", conn)#, params=('%' + term + '%'))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY unix DESC LIMIT 1000", conn)#, params=('%' + term + '%'))    
        df_sent = df.copy()
        df_sent.sort_values('unix', inplace=True)
        df_sent['date'] = pd.to_datetime(df_sent['unix'], unit='ms')
        df_sent.set_index('date', inplace=True)
        df_sent['sentiment_smoothed'] = df_sent['sentiment'].rolling(int(len(df)/10)).mean()


        resampled_df = df_resample(df_sent, time_bin)
        # df.dropna(inplace=True)

        X = resampled_df.index[-max_length:]
        Y = resampled_df.sentiment_smoothed.values[-max_length:]
        Y2 = resampled_df.volume.values[-max_length:]
        
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
    except Exception as e:
        print(e)


### LIVE WORD CLOUD UPDATE ###
@app.callback(Output('word-cloud-image', 'src'),
              [Input('get-word-cloud-button', 'n_clicks'),
               Input('sentiment-term', 'value')])
def update_word_cloud(n_clicks, term):
    # print('UHOHHHHH!!!!!!!!')
    df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%{}%' ORDER BY unix DESC LIMIT 5000".format(term), conn)#, params=('%' + term + '%'))
    img_file = get_word_cloud(df, term)

    return encode_image(img_file)
    return 'children'

### LIVE TWEET TABLE TITLE ###
@app.callback(Output('live-tweet-table-title', 'children'),
              [Input('sentiment-term', 'value')])
def update_table_title(term):
    if term:
        return f'Recent Tweets - {term}'
    else:
        return 'Recent Tweets'

## LIVE TWEEET TABLE UPDATE ###
@app.callback(Output('live-tweet-table', 'children'),
              [Input('live-tweet-table-update', 'n_intervals'),
               Input('sentiment-term', 'value')])        
def update_tweet_table(n, term): 
    MAX_ROWS=20
    df_table = df.copy()
    # df_table.sort_values('sentiment', ascending=True, inplace=True)
    df_table['date'] = pd.to_datetime(df_table['unix'], unit='ms')
    df_table['dealt'] = 'No'
    df_table['link'] = df_table['id'].apply(make_clickable)
    df_table = df_table[['date','tweet','sentiment', 'link', 'dealt']].iloc[:MAX_ROWS]

    return generate_tweet_table(df_table)

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
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=True)