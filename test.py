import pandas_datareader.data as web
import datetime
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime.now()

stock = 'TSLA'

df = web.DataReader(stock, 'google', start, end)