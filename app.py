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
plt.switch_backend('Agg')
from spacy.lang.en.stop_words import STOP_WORDS

import base64
from IPython.display import HTML
import time
import datetime
# Inspired by https://github.com/Sentdex/socialsentiment
# import ktrain

global refresh_time
refresh_time = 3

### SQL Connection ###
conn = sqlite3.connect('data/twitter_2.db', check_same_thread=False, timeout=25)
c = conn.cursor()





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

app.layout = html.Div(className='w3-row', children=[
    html.Div(className='container-fluid', children=[html.H1('Dashboard - Tweet Analysis', id='main-title')],
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
                                                                   options=[{'label':i, 'value':i} for i in ['5s', '10s', '30s', '60s']],
                                                                   value = '5s',
                                                                   style = {'color':colors['text']}),
                                                    html.Label('Maximum Number of Data: ', id='maximum-number-of-data', style={'color':colors['text']}),
                                                    dcc.RadioItems(id='sentiment-max-length',
                                                                   options=[{'label':i, 'value':i} for i in [100, 500, 1000, 2000, 5000]],
                                                                   value = 500,
                                                                   style = {'color':colors['text']})],
                                                    style={'textAlign':'left',
                                                           'padding':'5px',
                                                           'color':colors['text']}),

    # Live Sentiment Graph
    html.Div(className='row', children=[html.Div(dcc.Graph(id='live-sentiment-graph', animate=False), className='col s12 m6 l6')]),

    html.Hr(),
    html.Div(),
    html.Div(className='row', children=[html.Button('Generate Word Cloud!', id='get-word-cloud-button',
                                                    style={'display': 'inline-block',
                                                            'height': '50px',
                                                            'padding': '0 50px',
                                                            'margin': '0 20px 20px 0',
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
                                                            'box-sizing': 'border-box'}),
                                        # html.Div(id='word-cloud-loading-message', style={'textAlign':'center', 'color':colors['table-text'], 'padding':'10px'}),
                                        html.Div(className='container-fluid', children=[html.Div(html.Img(id='word-cloud-image', src='children', 
                                                                                                          style={'object-fit': 'cover',
                                                                                                                 'height':'600px', 
                                                                                                                 'width':'1500px',
                                                                                                                 }))])],
                                                    style={'textAlign': 'center'}),
    
                         
    html.Hr(),
    html.Div(className='container-fluid', children=[html.H2(id='live-tweet-table-title', 
                                                            style={'textAlign':'center',
                                                                   'color':colors['text'],
                                                                   'padding':'10px'})]),
    # LIVE TWEET TABLE
    html.Div(className='w3-row', children=[html.Div(id="live-tweet-table", className='w3-container w3-twothird'),
                                           html.Div(dcc.Graph(id='live-tweet-table-pie', animate=False), className='w3-container w3-third')]),
    
    html.Hr(),

    # LIVE FLAGGED TWEET TABLE
            
    html.Div(className='container-fluid', children=[html.H2(id='live-flagged-tweet-table-title', 
                                                            style={'textAlign':'center',
                                                                   'color':colors['ex-negative-sentiment'],
                                                                   'padding':'10px'})]),

    html.Div(className='row', children=[html.Button("FLAG ME!", id='live-flagged-tweet-table-update-button',
                                                    style={'display': 'inline-block',
                                                            'height': '50px',
                                                            'padding': '0 50px',
                                                            'margin': '0 20px 20px 0',
                                                            'color': colors['sl-negative-sentiment'], #'#555',
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
                                                    style={'textAlign': 'center'}),              

    html.Div(className='w3-row', children=[html.Div(id="live-flagged-tweet-table", className='w3-container w3-twothird'),
                                           html.Div('EMPTY SPACE',className='w3-container w3-third')]),
    
    html.Hr(),
    
    dcc.Interval(
        id='live-sentiment-graph-update',
        interval=refresh_time*1000, # in milliseconds
        n_intervals=0
    ),

    # dcc.Interval(
    #     id='live-word-cloud-update',
    #     interval=60*1000, # in milliseconds
    #     n_intervals=0
    # ),

    dcc.Interval(
        id='live-tweet-table-update',
        interval=(refresh_time+4)*1000, # in milliseconds
        n_intervals=0
    ),

    dcc.Interval(
        id='live-tweet-table-pie-update',
        interval=(refresh_time+5)*1000, # in milliseconds
        n_intervals=0
    ),
    # dcc.Interval(
    #     id='live-flagged-tweet-table-update',
    #     interval=10*1000, # in milliseconds
    #     n_intervals=0
    # ),

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

def get_word_cloud(dataframe, term=''):
    mask = np.array(Image.open('images/tweet_mask.jpg'))
    print('generating word cloud')
    stopword_list = list(STOP_WORDS) + list(string.punctuation) + [term] + ['good', 'bad', 'fuck', 'fucking', 'hate', 'best', 'awesome', 'great', 'horrible']

    wc_pos = WordCloud(mask=mask, background_color=colors['background'], stopwords=stopword_list,
                        max_font_size=256,
                        random_state=42, width=mask.shape[1]*1.2,
                        height=mask.shape[0]*1.2, color_func=similar_color_func_blue)
    wc_pos.generate(','.join(dataframe[dataframe['sentiment']>POS_THRESH]['clean']))

    wc_neg = WordCloud(mask=mask, background_color=colors['background'], stopwords=stopword_list,
                        max_font_size=256,
                        random_state=42, width=mask.shape[1]*1.2,
                        height=mask.shape[0]*1.2, color_func=similar_color_func_orange)
    wc_neg.generate(','.join(dataframe[dataframe['sentiment']<NEG_THRESH]['clean']))

    fig, axes = plt.subplots(ncols=2, figsize=(24,14))
    axes[0].axis('off')
    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].set_title('Positive Sentiment', fontdict={'fontsize': 25, 'fontweight': 'medium', 'color': 'white'})

    axes[1].axis('off')
    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].set_title('Negative Sentiment', fontdict={'fontsize': 25, 'fontweight': 'medium', 'color': 'white'})
    
    # timestamp = str(datetime.datetime.now())
    img_file = 'images/word_cloud.png'
    fig.savefig(img_file, facecolor=colors['background'], edgecolor=colors['background'])
    # print('creted word cloud files and about to save it!')
    plt.close('all')
    return img_file

def encode_image(image_file):
    encoded = base64.b64encode(open(image_file,'rb').read())
    print('Uploading the word cloud')
    return 'data:image/png;base64,{}'.format(encoded.decode())

def generate_tweet_table(dataframe):
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
    global refresh_time
    if max_length == 1000:
        refresh_time = 5
    elif max_length == 2000:
        refresh_time = 10
    elif max_length == 5000:
        refresh_time = 30

    try:
        global df
        if term:
            df = pd.read_sql(f"SELECT * FROM sentiment WHERE tweet LIKE '%{term}%' ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))
        else:
            df = pd.read_sql(f"SELECT * FROM sentiment ORDER BY unix DESC LIMIT {max_length}", conn)#, params=('%' + term + '%'))    
        df_sent = df.copy()

        df_sent.sort_values('unix', inplace=True)
        df_sent['date'] = pd.to_datetime(df_sent['unix'], unit='ms')
        df_sent.set_index('date', inplace=True)

        df_sent['sentiment_smoothed'] = df_sent['sentiment'].rolling(int(len(df)/10)).mean()


        resampled_df = df_resample(df_sent, time_bin)
        # df.dropna(inplace=True)

        X = resampled_df.index
        Y = resampled_df.sentiment_smoothed.values
        Y2 = resampled_df.volume.values
        
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
               Input('sentiment-term', 'value'),])
def update_word_cloud(n_clicks, term):
    print("You've just clicked the word cloud button!")
    # df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%{}%' ORDER BY unix DESC LIMIT 5000".format(sentiment_term), conn)#, params=('%' + term + '%'))
    if term:
        img_file = get_word_cloud(df, term)
    else:
        img_file = get_word_cloud(df)
    return encode_image(img_file)

## LIVE TWEEET TABLE UPDATE ###
@app.callback(Output('live-tweet-table', 'children'),
              Output('live-tweet-table-title', 'children'),
              [Input('live-tweet-table-update', 'n_intervals'),
               Input('sentiment-term', 'value')])        
def update_tweet_table(n, term):
    MAX_ROWS=15
    df_table = df.copy()
    # df_table.sort_values('sentiment', ascending=True, inplace=True)
    df_table['date'] = pd.to_datetime(df_table['unix'], unit='ms')
    df_table['link'] = df_table['id'].apply(make_clickable)
    df_table = df_table[['date','tweet','sentiment', 'link']].iloc[:MAX_ROWS]
    
    if term:
        title = f'Recent Tweets - {term}'
    else:
        title = 'Recent Tweets'

    return generate_tweet_table(df_table), title

## LIVE PIE UPDATE ###
@app.callback(Output('live-tweet-table-pie', 'figure'),
              [Input('live-tweet-table-pie-update', 'n_intervals'),
               Input('sentiment-term', 'value'),
               Input('sentiment-max-length', 'value')])        
def update_pie(n, term, max_num):
    df_pie = df.copy()

    labels = ['Positive', 'Negative', 'Neutral']
    pos = sum(df_pie['sentiment'] > 0)
    neg = sum(df_pie['sentiment'] < 0)
    neu = sum(df_pie['sentiment'] == 0)

    colors_pie = ['#58D68D', '#E74C3C', '#F7DC6F']

    pie = go.Pie(labels=labels, values = [pos, neg, neu],
                 hoverinfo='label+percent', textinfo='value',
                 textfont=dict(size=20, color=colors['table-text']),
                 marker=dict(colors=colors_pie,
                             line=dict(color=colors['background'], width=1)))

    return {"data": [pie], 'layout': go.Layout(title = f"Sentiment Distribution - {term}(n={max_num})",
                                             font={'color': '#566573'},
                                             plot_bgcolor = colors['background'],
                                             paper_bgcolor = colors['background'],
                                             showlegend = True)}

## LIVE FLAGGED TWEEET TABLE UPDATE ###
@app.callback(Output('live-flagged-tweet-table', 'children'),
              Output('live-flagged-tweet-table-title', 'children'),
              [Input('live-flagged-tweet-table-update-button', 'n_clicks'),
               Input('sentiment-term', 'value'),])
            #    Input('sentiment-term', 'value')])        
def update_flagged_tweet_table(n_clicks, term):
    print("You've just clicked 'FLAG ME' button!")
    MAX_ROWS=20
    if term:
        flagged_df = pd.read_sql("""SELECT * FROM flag 
                                    WHERE tweet LIKE '%{}%' AND dealt != 1
                                    ORDER BY unix DESC LIMIT 30""".format(term), conn)
    else:
        flagged_df = pd.read_sql("""SELECT * FROM flag 
                                    WHERE dealt != 1
                                    ORDER BY unix DESC LIMIT 30""".format(term), conn)

    flagged_df['date'] = pd.to_datetime(flagged_df['unix'], unit='ms')
    flagged_df['link'] = flagged_df['id'].apply(make_clickable)
    flagged_df = flagged_df[['date','tweet','sentiment', 'link', 'dealt']].iloc[:MAX_ROWS]
    
    if term:
        title = f'Recently Flagged Tweets - {term}'
    else:
        title = 'Recent Flagged Tweets'
    return generate_flagged_tweet_table(flagged_df), title

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
external_css = ["https://www.w3schools.com/w3css/4/w3.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=True)