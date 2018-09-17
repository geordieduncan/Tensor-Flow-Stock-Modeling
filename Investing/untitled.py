# Neural Network Test
import random
from sympy import Matrix
import numpy
import requests
from testset import learnset
from matplotlib import pyplot as plt
import time
import math


t1 = time.time()
api_key = 'KOGG0MERXF0070F0'
stock = 'MSFT'
statement = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+ stock + '&interval=1min&apikey='
k = 0.5

def f(a, b):
    return 0.1 * math.exp(1 - ((b / a) ** 2))

def sig(x):
    return numpy.exp(x) / (numpy.exp(x) + 1)


def choiceN(L):  # an average choice algorithm for numerical outputs
    return float(sum(L)) / float(len(L))


def choiceD(L):  # a maximum choice algorithm from descrete outputs
    return L.index(max(L))


def randMat(n, m, tstep):
    output = []
    for i in range(m):
        row = []
        for i in range(n):
            row.append(random.uniform(-1, 1) * tstep)
        output.append(row)
    return output


class NeuralNet:
    def __init__(self, n, midVars, Outs, stock):
        self.n = n  # length of inputs
        self.midVars = midVars  # list of layer sizes
        self.m = len(midVars)  # number of layers
        self.Outs = Outs  # length of number of outputs, can be 1 for numerical
        # creating the transfer matrices between hidden layers
        self.T = {}
        for i in range(self.m - 1):
            self.T[i + 1] = Matrix([[0.0] * self.midVars[i]] * self.midVars[i + 1])
        # creating intial and final transfer matrices
        self.T[0] = Matrix([[0.0] * self.n] * self.midVars[0])
        self.T[self.m] = Matrix([[0.0] * self.midVars[-1]] * self.Outs)

        # #statement = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+ stock + '&interval=1min&apikey='
        # statement = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey='
        # r = requests.get(statement + api_key)
        # JSON = r.json()
        # data = JSON['Time Series (Daily)']
        # dataset = []
        # Dset = []
        # for i in data.keys():
        #     Dset.append(data[i]['1. open'])
        # for i in Dset:
        #     dataset.append(float(i))
        # global learnset
        # learnset = []
        # for i in range(len(dataset) - (n + 1)):
        #     learnset.append((dataset[i:i + n], dataset[i + n]))

    def calc(self, I, T):
        Mat = Matrix(I)
        for i in range(self.m + 1):
            Mat = T[i] * Mat
            L = []
            for i in range(Mat.rows):
                L.append(Mat[i])
        return choiceN(L)  # numerical

    

    def propogate(self, trainSet):
        # back propogation for numerical outputs
        E1 = 0.0
        E2 = 0.0
        El = 1000.0
        EBase = El
        dT = {}
        for i in range(self.m - 1):
            dT[i + 1] = Matrix(randMat(self.midVars[i], self.midVars[i + 1], f(El, EBase)))
        dT[0] = Matrix(randMat(self.n, self.midVars[0], f(El, EBase)))
        dT[self.m] = Matrix(randMat(self.midVars[-1], self.Outs, f(El, EBase)))
        newT = {}
        for i in range(self.m + 1):
            newT[i] = self.T[i] + dT[i]
        for vals in trainSet:
            O1 = self.calc(vals[0], self.T)
            O2 = self.calc(vals[0], newT)
            E1 = abs(vals[1] - O1)
            E2 = abs(vals[1] - O2)  # does abs help????
            chfactor = sig(E1 - E2) * k

net = NeuralNet(10, [10, 10], 1, 'AAPL')

for i in range(150):
    net.propogate(learnset)
    for j in range(3):
        print ''
    print 'propogation {}'.format(i + 1)
    print net.calc(learnset[40][0], net.T)

actuals = []
guesses = []
for vals in learnset:
    actuals.append(vals[1])
    guesses.append(net.calc(vals[0], net.T))
plt.plot(actuals)
plt.plot(guesses, 'r')
plt.show()

t2 = time.time()
print t2 - t1


