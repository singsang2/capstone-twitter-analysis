import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas_datareader.data as web
import datetime
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime.now()

stock = 'TSLA'

df = web.DataReader(stock, 'yahoo', start, end)

# df = pd.read_csv('datasets/tweet_sentiments.csv', encoding='latin')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Dashboard - Tweet Analysis'),
    html.Div([
        html.Label(['Topics']),
        dcc.Dropdown(
            id='dropdown_topics',
            options=[
                     {'label': 'Microsoft Overall', 'value': 'value1'},
                     {'label': 'Xbox', 'value': 'Xbox'},
                     {'label': 'Office', 'value': 'Office'},
                     {'label': 'Azure', 'value': 'Azure'},
                     {'label': 'Team', 'value': 'Team'},
                     {'label': 'Cloud', 'value': 'Cloud'}
            ],
            value = 'dropdown #1',
            multi=True,
            clearable=False,
            style={"width": "40%"})
    ]),

    html.Div([
        dcc.Graph(id='example-stock',
                  figure = {
                      'data': [
                          {'x':df.index, 'y':df.Close, 'type': 'line', 'name': stock},
                      ],
                      'layout': {
                          'title': stock
                      }
                  }
                 )
    ]),
    html.Div([
        html.Label('Input Field'),
        dcc.Input(id='input_num', value='Enter something', type='text'),
        html.Div(id='output_num'),
    ])
    
])

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