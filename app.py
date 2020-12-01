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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# # df = pd.read_csv('datasets/tweet_sentiments.csv', encoding='latin')


#### STYLE ####
colors = {'background': '#111111', 'text':'#7FDBFF'}



#### APP ####
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('Dashboard - Tweet Analysis', style={'textAlign':'center',
                                                 'color':colors['text']}),
    # Live Graph Test
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='graph-update',
        interval=1000, # in milliseconds
        n_intervals=0
    ),

    html.Div([
        html.Label(['Companies']),
        dcc.Dropdown(
            id='stock-companies',
            options=[
                     {'label': 'TSLA', 'value': 'TSLA'},
                     {'label': 'AAPL', 'value': 'AAPL'},
                     {'label': 'MSFT', 'value': 'MSFT'},
                     {'label': 'YHOO', 'value': 'YHOO'},
                     {'label': 'GOOGL', 'value': 'GOOGL'},
            ],
            value = 'Select Companies',
            multi=True,
            clearable=False,
            style={"width": "40%"}),
        dcc.RadioItems(id='starting-year',
                   options=[
                       {'label': '2015', 'value': 2015},
                       {'label': '2016', 'value': 2016},
                       {'label': '2017', 'value': 2017},
                       {'label': '2018', 'value': 2018},
                       {'label': '2019', 'value': 2019},],
                    value = 2015,
                    style = {'color':colors['text']}
                    )
    ]),


    dcc.Input(id='input', value='', type='text'),
    html.Div(id='output-graph'),

    html.Div([
        html.Label('Input Field'),
        dcc.Input(id='input_num', value='Enter something', type='text'),
        html.Div(id='output_num'),
    ]),
    
],
    # Defines overall style
    style = {'backgroundColor': colors['background']}
)

# @app.callback(Output('live-graph', 'figure'),
#               Input('graph-update', 'n_interval'))
# def update_graph(n):
#     global X
#     global Y
#     X.append(X[-1] + 1)
#     Y.append(Y[-1] + Y[-1]*random.uniform(-0.1, 0.1))

#     data = go.Scatter(
#         X = list(X),
#         y = list(Y),
#         name = 'Scatter',
#         mode = 'lines+markers'
#     )

#     return {'data': [data],
#             'layout': go.Layout(xaxis = dict(range=[min(X), max(X)]),
#                                 yaxis = dict(range=[min(Y), max(Y)]))}
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

if __name__ == '__main__':
    app.run_server(debug=True)