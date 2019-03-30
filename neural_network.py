import numpy as np
from os import listdir
from os.path import isfile, join
import math

def func1(dot):
    return math.log(1 + math.exp(dot))

def func1der(dot):
    return math.exp(dot)/(1 + math.exp(dot))

def func2(dot):
    return (math.exp(dot) - math.exp(-dot))/(math.exp(dot) + math.exp(-dot))

def func2der(dot):
    return 1 - math.pow(func2(dot),2)

def func3(dot):
    return math.cos(dot)

def func3der(dot):
    return -math.sin(dot)

#use by default ax=1, when the array is 2D
#use ax=0 when the array is 1D
def softmax( x, ax=1 ):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )

'''def softmax(W, z, k):
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
'''
class NeuNet():

    def __init__(self, hidlen, outlen, func, der):
        self.hidlen = hidlen # for the bias term
        self.outlen = outlen
        #self.hidden = np.zeros((1,hidlen))
        self.activation = func
        self.activation_der = der
        self.train = []
        self.train_data = []
        self.test = []
        self.inlen = 0
        self.w0 = None
    def readMnistData(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        print(files)
        self.train = [f for f in files if 'train' in f]
        self.test = [f for f in files if 'test' in f]
        print()
        print(self.test)
        print()
        print(self.train)
        #lines = []
        for f in self.train[:1]: # read only the first training
            with open(join(path,f), 'r') as fl:
                for line in fl:
                    self.train_data.append(list(map(int, line.split(' '))))
                    #lines.append(line.split(' '))
        #print(lines[3])

        self.train_data = np.array([np.array(x) for x in self.train_data])
        self.train_data = self.train_data.astype(float) / 255 # normalize the data
        self.train_data = np.hstack((np.ones((self.train_data.shape[0], 1)), self.train_data))  # +1 col at the begining of the array for the bias term
        print(self.train_data.shape)
        # now that we have read the training data, we can initialize the input layer
        self.inlen = len(self.train_data[0])
        print('inlen: ' + str(self.inlen))
        self.w0 = np.random.rand(self.hidlen,self.inlen) # the 2D array that contains the weights of the first part of the neural network
        self.w1 = np.random.rand(self.outlen,self.hidlen + 1) # the 2D array that contains the weights of the second part of the neural network + 1 for the bias

    def feed_forward(self):
        j = 0
        for data in self.train_data: # for each training data
            input_vec = np.array(data, ndmin=2)
            #input_vec = data
            #input_vec = np.hstack((input_vec, np.array([1], ndmin=2)))# +1 for the bias term
            print(input_vec.shape)
            print(self.w0.shape)
            temp = input_vec.dot(self.w0.transpose())
            print(temp.shape)
            #hidden_layer_out = np.array(list(map(self.activation,temp)),ndmin=2)
            hidden_leyer_out = []
            print(temp)
            for i in range(temp.shape[1]):
                hidden_leyer_out.append(self.activation(temp[0][i]))
            hidden_leyer_out = np.array(hidden_leyer_out, ndmin=2)
            hidden_leyer_out = np.hstack((hidden_leyer_out, np.array([1], ndmin=2)))# +1 for the bias term
            print(hidden_leyer_out.shape)
            temp = hidden_leyer_out.dot(self.w1.transpose())
            print(temp)
            print(temp.shape)
            # now temp contains the dot products of the second half
            # and the only thing that is left is to give it as
            # input to the softmax function
            final_out = softmax(temp)
            print(final_out)
            if j == 5:
                break
            j += 1
        
if __name__ == '__main__':
    net = NeuNet(100,10, func3, func3der)
    #print(net.w0)
    #print(net.w1)

    x = 8.323

    '''print(func1(x))
    print(func1der(x))
    print(func2(x))
    print(func2der(x))
    print(func3(x))
    print(func3der(x))
    '''
    #print(softmax(np.array( [ [10,20,30,40], [20,50,45,45], [983,39,57,752], [574,575,597,525] ] )))
    net.readMnistData('/home/marios/Downloads/mnistdata')
    net.feed_forward()
