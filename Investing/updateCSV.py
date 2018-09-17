import requests
import csv
import pprint
import time
n = 10
stocks = ['AAPL', 'BABA', 'LMT', 'MSFT', 'T', 'TGT', 'WMT']

for stock in stocks:
    api_key = 'H89PZBIJISUP58C8'
    statement = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + stock + '&interval=5min&apikey=' + api_key
    r = requests.get(statement)
    JSON = r.json()
    data = JSON['Time Series (5min)']
    dataset = []
    Dset = []
    for i in data.keys():
        Dset.append(data[i]['1. open'])
    for i in Dset:
        dataset.append(float(i))
    global learnset
    learnset = []
    for i in range(len(dataset) - (n + 1)):
        learnset.append((dataset[i:i + n], dataset[i + n]))
    file = open(stock + '.csv', 'w')
    for i in learnset:
        line = str(i[1]).strip(']').strip('[') + ',' + str(i[0][::-1]).strip(']').strip('[') + '\n'
        line = line.strip(']').strip('[')
        file.write(line) 
    file.close
    time.sleep(12)


