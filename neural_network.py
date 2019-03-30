import numpy as np
from os import listdir
from os.path import isfile, join
import math

def func1(dot):
    return math.log(1 + math.exp(dot))

def func1der(dot):
    return math.exp(dot)/(1 + math.exp(dot))

def func2(dot):
    return (math.exp(dot) - math.exp(-dot))/(math.exp(dot) - math.exp(-dot))

def func2der(dot):
    return 1 - math.pow(func2(dot),2)

def func3(dot):
    return math.cos(dot)

def func3der(dot):
    return -math.sin(dot)


def softmax(W, z, k):
    temp = 0
    
    dotk = 0
    for i in range(W.shape[0]):
        dot = 0
        for j in range(len(z)):
            dot += W[i][j]*z[j]
        temp += math.exp(dot)
        if i == k:
            dotk = dot
    
    return math.exp(dotk)/temp

class NeuNet():

    def __init__(self, inlen, hidlen, outlen, func, der):
        self.inlen = inlen + 1 # for the bias term
        self.hidlen = hidlen + 1 # for the bias term
        self.outlen = outlen
        self.w0 = np.random.rand(self.hidlen, self.inlen)
        self.w1 = np.random.rand(self.outlen, self.hidlen)
        self.hidden = np.zeros((1,hidlen))
        self.train = []
        self.test = []
        self.indata = None

        
    def readMnistData(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        print(files)
        self.train = [f for f in files if 'train' in f]
        self.test = [f for f in files if 'test' in f]
        print()
        print(self.test)
        print()
        print(self.train)
        lines = []
        for f in self.train[:1]:
            with open(join(path,f), 'r') as fl:
                for line in fl:
                    lines.append(list(map(int, line.split(' '))))
                    #lines.append(line.split(' '))
        print(lines[0])
        
if __name__ == '__main__':
    net = NeuNet(3,3,2, func1, func1der)
    print(net.w0)
    print(net.w1)

    x = 8.323

    print(func1(x))
    print(func1der(x))
    print(func2(x))
    print(func2der(x))
    print(func3(x))
    print(func3der(x))
    
    print(softmax(net.w0, [i for i in range(len(net.w0[0]))],1))
    net.readMnistData('/home/marios/Downloads/mnistdata')
