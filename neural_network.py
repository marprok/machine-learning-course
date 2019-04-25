import numpy as np
from os import listdir
from os.path import isfile, join
import math
from random import shuffle

def func1(dot):
    return math.log(1 + math.exp(dot))

def func1der(dot):
    return math.exp(dot)/(1 + math.exp(dot))

def func2(dot):
    return (math.exp(dot) - math.exp(-dot))/(math.exp(dot) + math.exp(-dot))

def func2der(dot):
    return 1 - math.pow(func2(dot),2)

def func3(vertex):
    return np.cos(vertex)

def func3der(vertex):
    return -1*np.sin(vertex)

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

    def __init__(self, hidlen, outlen, func, der, batch = 100):
        self.hidlen = hidlen # for the bias term
        self.outlen = outlen
        #self.hidden = np.zeros((1,hidlen))
        self.activation = func
        self.activation_der = der
        self.train = []
        self.train_data = []
        self.train_1hv = []

        self.test = []
        self.test_data = []
        self.test_1hv = []
        self.inlen = 0
        self.w0 = None
        self.batch = batch
        self.epochs = None

    def shuffle_data(self, data, hotvec):
        # zip the one hot vector and the training data together
        temp = list(zip(data, hotvec))
        # shuffle them
        shuffle(temp)
        # unzip
        return zip(*temp)
    
    def train_net(self, epochs):
        self.epochs = epochs
        prevcost = 0
        for i in range(epochs):
            # first we shuffle the training data
            if i % 10 == 0:
                print('shuffling data')
                self.train_data, self.train_1hv = self.shuffle_data(self.train_data, self.train_1hv)

            if self.batch > len(self.train_data):
                print("ERROR: batch size is greater than the total number of training data")
                return None

            begin, end = 0, self.batch
            cost = None

            while begin < end:
                data_batch = self.train_data[begin:end]
                data_batch = np.array([x for x in data_batch])
                hot_vec = np.array(self.train_1hv[begin:end])

                # print('begin: ' + str(begin) +'\nend: ' + str(end))
                cost, hidden_out, final_out = self.feed_forward(data_batch, hot_vec)
                dw0, dw1 = self.computeGrads(data_batch, hidden_out, final_out, hot_vec, 0.01)

                self.w0 += 0.001*dw0
                self.w1 += 0.001*dw1

                begin = end
                end += self.batch
                if end > len(self.train_data):
                    end = len(self.train_data)

            #if max_cost < cost:

             #   max_cost = cost
            #diff = np.abs(np.abs(cost) - np.abs(prevcost))
            #print(diff)
            #if cost > -3.0:
              #  print('final cost: ', cost, ' epoch: ', i)
             #   break
            #prevcost = cost
            print('cost: ',cost,' epoch = ', i)

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    def readCIFAR10Data(self, file):

        files = [f for f in listdir(file) if isfile(join(file, f))]
        #print(files)
        self.train = [file + '/' + f for f in files if 'data_batch' in f]
        self.test = [file + '/' + f for f in files if 'test_batch' in f]

        #print(self.train)
        #print(self.test)

        for f in self.train:
            dict = self.unpickle(f)
            temp = dict['data']
            for line in temp:
                #print(line)
                self.train_data.append(line)
            #print(dict['data'])
            #print(dict['labels'])

            for label in dict['labels']:
                self.train_1hv.append(
                       np.array([1 if i == label else 0 for i in range(10)]))  # create the one hot vector


        self.train_1hv = np.array([x for x in self.train_1hv])
        print('train y', self.train_1hv.shape)
        self.train_data = np.array([np.array(x) for x in self.train_data])
        self.train_data = self.train_data.astype(float) / 255  # normalize the data
        # the last column of the data matrix is the bias column
        self.train_data = np.hstack((np.ones((self.train_data.shape[0], 1)), self.train_data))
        print('train data', self.train_data.shape)
        # now that we have read the training data, we can initialize the input layer
        self.inlen = len(self.train_data[0])
        # print('inlen: ' + str(self.inlen))
        self.w0 = np.random.normal(0, 1 / np.sqrt(self.inlen), (
        self.hidlen, self.inlen))  # the 2D array that contains the weights of the first part of the neural network
        self.w1 = np.random.rand(self.outlen,
                                 self.hidlen + 1)  # the 2D array that contains the weights of the second part of the neural network + 1 for the bias


        for f in self.test:
            dict = self.unpickle(f)
            temp = dict['data']
            for line in temp:
                #print(line)
                self.test_data.append(line)
            #print(dict['data'])
            #print(dict['labels'])

            for label in dict['labels']:
                self.test_1hv.append(
                       np.array([1 if i == label else 0 for i in range(10)]))  # create the one hot vector
                #print(np.array([1 if i == label else 0 for i in range(10)]))
        self.test_1hv = np.array([x for x in self.test_1hv])
        self.test_data = np.array([np.array(x) for x in self.test_data])
        self.test_data = self.test_data.astype(float) / 255  # normalize the data
        # the last column of the data matrix is the bias column
        self.test_data = np.hstack((np.ones((self.test_data.shape[0], 1)),
                                    self.test_data))  # +1 col at the start of the array for the bias term
        print('test data', self.test_data.shape)
        #print(self.w0.shape)
        #print(self.w1.shape)

        
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

        for f in self.test[:]:  # read only the first training
            num = int(f.split('test')[1][0])

            with open(join(path, f), 'r') as fl:
                for line in fl:
                    self.test_data.append(list(map(int, line.split(' '))))
                    self.test_1hv.append(
                        np.array([1 if i == num else 0 for i in range(10)]))  # create the one hot vector
                    # lines.append(line.split(' '))
            # print(lines[3])
        self.test_1hv = np.array([x for x in self.test_1hv])
        # print(self.train_1hv.shape)
        # print(self.train_1hv)

        self.test_data = np.array([np.array(x) for x in self.test_data])
        self.test_data = self.test_data.astype(float) / 255  # normalize the data
        # the last column of the data matrix is the bias column
        self.test_data = np.hstack((np.ones((self.test_data.shape[0], 1)),
                                     self.test_data))  # +1 col at the start of the array for the bias term
        print(self.test_data.shape)

        
        for f in self.train[:]: # read only the first training
            num = int(f.split('train')[1][0])
            
            #print(self.train_1hv)
            #print('the number is ' + str(num))
            with open(join(path,f), 'r') as fl:
                for line in fl:
                    self.train_data.append(list(map(int, line.split(' '))))
                    self.train_1hv.append(np.array([1 if i == num else 0 for i in range(10)])) # create the one hot vector
                    #lines.append(line.split(' '))
        #print(lines[3])
        self.train_1hv = np.array([x for x in self.train_1hv])
        #print(self.train_1hv.shape)
        #print(self.train_1hv)
        
        self.train_data = np.array([np.array(x) for x in self.train_data])
        self.train_data = self.train_data.astype(float) / 255 # normalize the data
        # the last column of the data matrix is the bias column
        self.train_data = np.hstack((np.ones((self.train_data.shape[0], 1)), self.train_data))  # +1 col at the start of the array for the bias term
        print(self.train_data.shape)
        # now that we have read the training data, we can initialize the input layer
        self.inlen = len(self.train_data[0])
        #print('inlen: ' + str(self.inlen))
        self.w0 = np.random.normal(0,1/np.sqrt(self.inlen),(self.hidlen,self.inlen)) # the 2D array that contains the weights of the first part of the neural network
        self.w1 = np.random.rand(self.outlen,self.hidlen + 1) # the 2D array that contains the weights of the second part of the neural network + 1 for the bias

        # gradient check
        #self.gradCheck(self.train_data, self.train_1hv)

    def computeGrads(self, data_batch, hidden_out, output, onehotvec, lamda = 0):
        # Calculate the gradient for w1
        grad_w1 = (onehotvec - output).T.dot(hidden_out) - lamda * self.w1
        
        # remove the bias term
        w1_temp = np.copy(self.w1[:, 1:])

        der = self.activation_der(data_batch.dot(self.w0.T))
        
        temp = (onehotvec - output).dot(w1_temp) * der
        
        # Calculate the gradient for w0
        grad_w0 = temp.T.dot(data_batch) - lamda*self.w0
        
        
        return grad_w0, grad_w1


    def feed_forward(self, data_batch, hot_vec):

        temp0 = data_batch.dot(self.w0.T)

        hidden_leyer_out = self.activation(temp0)
        hidden_leyer_out = np.hstack((np.ones((hidden_leyer_out.shape[0], 1)),
                                      hidden_leyer_out))  # +1 col at the start of the array for the bias term

        temp1 = hidden_leyer_out.dot(self.w1.T)

        final_out = softmax(temp1)


        # print('error: ',error)
        return self.calculate_cost(final_out, hot_vec), hidden_leyer_out, final_out

    def calculate_cost(self, y, t):
        #print(y)
        y = np.log(y)
        #print(y)
        l = 0.01
        return np.sum(y*(t)) - l/2 * (np.linalg.norm(self.w0, 'fro')**2 + np.linalg.norm(self.w1, 'fro')**2)

    def gradCheck(self, xarg, targ):
        w0 = self.w0
        w1 = self.w1
        e = 1e-6
        samples = np.random.randint(xarg.shape[0], size=5)
        x = np.array(xarg[samples, :])
        t = np.array(targ[samples, :])

        w0t = self.w0

        cost, hidden_out, final_out = self.feed_forward(x, t)
        gradw0, gradw1 = self.computeGrads(x, hidden_out, final_out, t, 0.01)

        # numeric stores all numerical gradients
        numeric = np.zeros(gradw0.shape)
        print(gradw0.shape, gradw1.shape, w0.shape, w1.shape)
        for k in range(numeric.shape[0]):
            for d in range(numeric.shape[1]):
                w_tmp = np.copy(w0)
                w_tmp[k, d] += e
                self.w0 = w_tmp;
                costeplus, hidden_out, final_out = self.feed_forward(x, t)

                w_tmp = np.copy(w0)
                w_tmp[k, d] -= e
                self.w0 = w_tmp;
                costeminus, hidden_out, final_out = self.feed_forward(x, t)

                numeric[k, d] = (costeplus - costeminus) / (2 * e)

        # Absolute norm
        print("For W0, the maximum difference between the numerical gradient and the one we found is: ", np.max(np.abs(gradw0 - numeric)))
        self.w0 = w0t
        numeric = np.zeros(gradw1.shape)

        for k in range(numeric.shape[0]):
            for d in range(numeric.shape[1]):
                # Calculate W1 gradient
                w_tmp = np.copy(w1)
                w_tmp[k, d] += e
                self.w1 = w_tmp;
                costeplus, hidden_out, final_out = self.feed_forward(x, t)

                w_tmp = np.copy(w1)
                w_tmp[k, d] -= e

                self.w1 = w_tmp;
                costminus, hidden_out, final_out = self.feed_forward(x, t)

                numeric[k, d] = (costeplus - costminus) / (2 * e)

        print("For W1, the maximum difference between the numerical gradient and the one we found is: ", np.max(np.abs(gradw1 - numeric)))

    def testNet(self):
        correct = 0
        # first we shuffle the training data
        self.test_data, self.test_1hv = self.shuffle_data(self.test_data, self.test_1hv)
        epochs = 1
        batch = 1
        for i in range(epochs):
            # first we shuffle the training data
            self.test_data, self.test_1hv = self.shuffle_data(self.test_data, self.test_1hv)
            if batch > len(self.test_data):
                print("ERROR: batch size is greater than the total number of training data")
                return None

            begin = 0
            end = batch
            cost = None

            while begin < end:
                data_batch = self.test_data[begin:end]
                data_batch = np.array([x for x in data_batch])
                hot_vec = np.array(self.test_1hv[begin:end])

                # print('begin: ' + str(begin) +'\nend: ' + str(end))
                cost, hidden_out, final_out = self.feed_forward(data_batch, hot_vec)
                dw0, dw1 = self.computeGrads(data_batch, hidden_out, final_out, hot_vec, 0.01)

                expected = np.argmax(hot_vec[0])
                predicted = np.argmax(final_out[0])

                print(expected,' ', predicted)
                #print(predicted)

                if expected == predicted:
                    correct += 1

                begin = end
                end += batch
                if end > len(self.test_data):
                    end = len(self.test_data)

        print(float(correct)/len(self.test_data))

if __name__ == '__main__':
    net = NeuNet(100,10, func3, func3der, 50)
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
    #net.readCIFAR10Data('C:/Users/Alexandros/Downloads/cifar-10-python/cifar-10-batches-py')
    #net.train_net(100)
    net.readMnistData('C:/Users/Alexandros/Downloads/mnistdata')
    net.train_net(100)

    net.testNet()

