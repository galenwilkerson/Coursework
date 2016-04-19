# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 01:43:30 2016

@author: username
"""
from datetime import datetime
import pandas as pd
#import pandas_datareader.data as web
import requests

from pandas.io.data import DataReader
from datetime import datetime

def get_intraday_data(symbol, interval_seconds=301, num_days=10):
    # Specify URL string based on function inputs.
    url_string = 'http://www.google.com/finance/getprices?q={0}'.format(symbol.upper())
    url_string += "&i={0}&p={1}d&f=d,o,h,l,c,v".format(interval_seconds,num_days)

    # Request the text, and split by each line
    r = requests.get(url_string).text.split()

    # Split each line by a comma, starting at the 8th line
    r = [line.split(',') for line in r[7:]]

    # Save data in Pandas DataFrame
    df = pd.DataFrame(r, columns=['Datetime','Close','High','Low','Open','Volume'])

    # Convert UNIX to Datetime format
    df['Datetime'] = df['Datetime'].apply(lambda x: datetime.fromtimestamp(int(x[1:])))

    return df
    
#2016-03-21
ibm = DataReader('IBM',  'yahoo', datetime(2016,3,21), datetime(2016,3,21))
#print(ibm['Adj Close'])
print ibm

df = get_intraday_data("AAPL", interval_seconds=301, num_days = 1)
print df