import TensorTest2
from Tkinter import *

stocks = ['AAPL', 'BABA', 'LMT', 'MSFT', 'T', 'TGT', 'WMT']
flist = []
wordlist = []
if __name__ == "__main__":
    for stock in stocks:
        try:
            f = TensorTest2.BuildTester(stock)[0]
            mse = TensorTest2.BuildTester(stock)[1]
            flist.append(f)
            words = stock + ":  " + str(round(f, 4),) + "(MSE: {})".format(mse)
            wordlist.append(words)
        except IOError:
            pass
w = {}

root = Tk()
for i in range(len(wordlist)):
    w[i] = Text(root, height=5, width=100)
    if flist[i] >= 0:
        w[i].tag_config("n", background="white", foreground="green")
    else:
        w[i].tag_config("n", background="white", foreground="red")
    w[i].tag_configure('big', font=('Verdana', 20, 'bold'))
    w[i].insert(END, wordlist[i], ("n", 'big'))
    w[i].pack()
root.mainloop()


def update():
    execfile("updateCSV.py")
