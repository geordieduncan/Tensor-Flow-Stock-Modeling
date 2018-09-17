import TensorTest2
stocks = ['AAPL', 'BABA', 'LMT', 'MSFT', 'T', 'TGT', 'WMT']

if __name__ == "__init__":
    for stock in stocks:
        TensorTest2.BuildTester(stock)


def update():
    execfile("updateCSV.py")
