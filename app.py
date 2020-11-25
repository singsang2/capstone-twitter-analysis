import pandas as pd

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

X = deque(maxlen=20)
Y = deque(maxlen=20)
X.append(1)
Y.append(1)

# # df = pd.read_csv('datasets/tweet_sentiments.csv', encoding='latin')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Dashboard - Tweet Analysis'),

    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='graph-update',
        interval=1000, # in milliseconds
    ),

    html.Div([
        html.Label(['Companies']),
        dcc.Dropdown(
            id='stock_companies',
            options=[
                     {'label': 'TSLA', 'value': 'TSLA'},
                     {'label': 'AAPL', 'value': 'AAPL'},
                     {'label': 'MSFT', 'value': 'MSFT'},
                     {'label': 'YHOO', 'value': 'YHOO'},
                     {'label': 'GOOGL', 'value': 'GOOGL'},
            ],
            value = 'dropdown #1',
            multi=True,
            clearable=False,
            style={"width": "40%"})
    ]),

    dcc.Input(id='input', value='', type='text'),
    html.Div(id='output-graph'),
    # html.Div([
    #     dcc.Graph(id='example-stock',
    #               figure = {
    #                   'data': [
    #                       {'x':df.index, 'y':df.Close, 'type': 'line', 'name': stock},
    #                   ],
    #                   'layout': {
    #                       'title': stock
    #                   }
    #               }
    #              )
    # ]),

    html.Div([
        html.Label('Input Field'),
        dcc.Input(id='input_num', value='Enter something', type='text'),
        html.Div(id='output_num'),
    ])
    
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_interval')]
)
def update_graph():
    global X
    global Y
    X.append(X[-1] + 1)
    Y.append(Y[-1] + Y[-1]*random.uniform(-0.1, 0.1))

    data = go.Scatter(
        X = list(X),
        y = list(Y),
        name = 'Scatter',
        mode = 'lines+markers'
    )

    return {'data': [data],
            'layout': go.Layout(xaxis = dict(range=[min(X), max(X)]),
                                yaxis = dict(range=[min(Y), max(Y)]))}
@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
)
def update_graph(input_data):
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime.now()

    df = web.DataReader(input_data, 'yahoo', start, end)

    return dcc.Graph(id='example-stock',
              figure = {
                'data': [
                    {'x':df.index, 'y':df.Close, 'type': 'line', 'name': input_data},
                ],
                'layout': {
                    'title': input_data
                }
              }
              )
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

if __name__ == '__main__':
    app.run_server(debug=True)