import numpy as np
from os import listdir
from os.path import isfile, join
from random import shuffle

# Marios Prokopakis p3150141


# The activation functions as well as their derivatives
def func1(dot):
    return np.log(1 + np.exp(dot))

def func1der(dot):
    return np.exp(dot)/(1 + np.exp(dot))

def func2(dot):
    return (np.exp(dot) - np.exp(-dot))/(np.exp(dot) + np.exp(-dot))

def func2der(dot):
    return 1 - ((np.exp(dot) - np.exp(-dot))**2) / ((np.exp(dot) + np.exp(-dot))**2)

def func3(dot):
    return np.cos(dot)

def func3der(dot):
    return -1*np.sin(dot)
# Taken from the labs
def softmax( x, ax=1 ):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )

# This class represents a neural network
class NeuNet():

    def __init__(self, hidlen, outlen, afunc, der, batch = 100, lm = 0.01):
        self.hidlen = hidlen # for the bias term
        self.outlen = outlen
        self.activation = afunc
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
        self.l = lm
    # This method shuffles the data and their corresponding 1 hot vectors
    def shuffle_data(self, data, hotvec):
        # zip the one hot vector and the training data together
        temp = list(zip(data, hotvec))
        # shuffle them
        shuffle(temp)
        # unzip
        return zip(*temp)

    # This method trains the network
    def train_net(self, epochs):
        self.epochs = epochs
        if False:
            self.gradCheck(self.train_data, self.train_1hv)
            return

        for i in range(epochs):
            # first we shuffle the training data
            if i % 10 == 0:
               #print('shuffling data')
                self.train_data, self.train_1hv = self.shuffle_data(self.train_data, self.train_1hv)

            if self.batch > len(self.train_data):
                print("ERROR: batch size is greater than the total number of training data")
                return None

            begin, end = 0, self.batch
            cost = None
            # while we have batches left
            while begin < end:
                data_batch = self.train_data[begin:end]
                data_batch = np.array([x for x in data_batch])
                hot_vec = np.array(self.train_1hv[begin:end])

                # print('begin: ' + str(begin) +'\nend: ' + str(end))
                cost, hidden_out, final_out = self.feed_forward(data_batch, hot_vec)
                dw0, dw1 = self.computeGrads(data_batch, hidden_out, final_out, hot_vec)
                # update the weights
                self.w0 += 0.001*dw0
                self.w1 += 0.001*dw1

                begin = end
                end += self.batch
                if end > len(self.train_data):
                    end = len(self.train_data)


    # Taken from the cifar site
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    # This method reads the cifar data set
    def readCIFAR10Data(self, file):

        files = [f for f in listdir(file) if isfile(join(file, f))]
        #print(files)
        self.train = [file + '/' + f for f in files if 'data_batch' in f]
        self.test = [file + '/' + f for f in files if 'test_batch' in f]

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
        self.w0 = np.random.normal(0, 1 / np.sqrt(self.inlen), (
            self.hidlen, self.inlen))  # the 2D array that contains the weights of the first part of the neural network

        self.w1 = np.random.normal(0, 1 / np.sqrt(self.hidlen + 1), (
            self.outlen,
            self.hidlen + 1))  # the 2D array that contains the weights of the second part of the neural network

        for f in self.test:
            dict = self.unpickle(f)
            temp = dict['data']
            for line in temp:
                self.test_data.append(line)

            for label in dict['labels']:
                self.test_1hv.append(
                       np.array([1 if i == label else 0 for i in range(10)]))  # create the one hot vector
        self.test_1hv = np.array([x for x in self.test_1hv])
        self.test_data = np.array([np.array(x) for x in self.test_data])
        self.test_data = self.test_data.astype(float) / 255  # normalize the data
        # the last column of the data matrix is the bias column
        self.test_data = np.hstack((np.ones((self.test_data.shape[0], 1)),
                                    self.test_data))  # +1 col at the start of the array for the bias term
        print('test data', self.test_data.shape)
        # gradient check(uncomment)
        # self.gradCheck(self.train_data, self.train_1hv)

    # This method reads the mnist data set
    def readMnistData(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        print(files)
        self.train = [f for f in files if 'train' in f]
        self.test = [f for f in files if 'test' in f]
        print()
        print(self.test)
        print()
        print(self.train)

        for f in self.test[:]:  # read only the first training
            num = int(f.split('test')[1][0])

            with open(join(path, f), 'r') as fl:
                for line in fl:
                    self.test_data.append(list(map(int, line.split(' '))))
                    self.test_1hv.append(
                        np.array([1 if i == num else 0 for i in range(10)]))  # create the one hot vector

        self.test_1hv = np.array([x for x in self.test_1hv])


        self.test_data = np.array([np.array(x) for x in self.test_data])
        self.test_data = self.test_data.astype(float) / 255  # normalize the data
        # the last column of the data matrix is the bias column
        self.test_data = np.hstack((np.ones((self.test_data.shape[0], 1)),
                                     self.test_data))  # +1 col at the start of the array for the bias term
        print(self.test_data.shape)

        
        for f in self.train[:]: # read only the first training
            num = int(f.split('train')[1][0])

            with open(join(path,f), 'r') as fl:
                for line in fl:
                    self.train_data.append(list(map(int, line.split(' '))))
                    self.train_1hv.append(np.array([1 if i == num else 0 for i in range(10)])) # create the one hot vector
        self.train_1hv = np.array([x for x in self.train_1hv])

        self.train_data = np.array([np.array(x) for x in self.train_data])
        self.train_data = self.train_data.astype(float) / 255 # normalize the data
        # the last column of the data matrix is the bias column
        self.train_data = np.hstack((np.ones((self.train_data.shape[0], 1)), self.train_data))  # +1 col at the start of the array for the bias term
        print(self.train_data.shape)
        # now that we have read the training data, we can initialize the input layer
        self.inlen = len(self.train_data[0])
        self.w0 = np.random.normal(0,1/np.sqrt(self.inlen),(self.hidlen,self.inlen)) # the 2D array that contains the weights of the first part of the neural network
        self.w1 = np.random.normal(0, 1 / np.sqrt(self.hidlen + 1), (
            self.outlen,
            self.hidlen + 1))  # the 2D array that contains the weights of the second part of the neural network
        # gradient check(uncomment)
        #self.gradCheck(self.train_data, self.train_1hv)

    # This method calculates the gradients that will be used to update th weights
    def computeGrads(self, data_batch, hidden_out, output, onehotvec):
        # Calculate the gradient for w1
        grad_w1 = (onehotvec - output).T.dot(hidden_out) - self.l * self.w1
        
        # remove the bias term
        w1_temp = np.copy(self.w1[:, 1:])

        der = self.activation_der(data_batch.dot(self.w0.T))
        
        temp = (onehotvec - output).dot(w1_temp) * der
        
        # Calculate the gradient for w0
        grad_w0 = temp.T.dot(data_batch) - self.l*self.w0
        
        
        return grad_w0, grad_w1

    # this method feeds the data to our network and computes the cost, the output of the hidden layer
    # as well as the output of our network
    def feed_forward(self, data_batch, hot_vec):

        temp0 = data_batch.dot(self.w0.T)

        hidden_leyer_out = self.activation(temp0)
        hidden_leyer_out = np.hstack((np.ones((hidden_leyer_out.shape[0], 1)),
                                      hidden_leyer_out))  # +1 col at the start of the array for the bias term

        temp1 = hidden_leyer_out.dot(self.w1.T)

        final_out = softmax(temp1)

        return self.calculate_cost(final_out, hot_vec), hidden_leyer_out, final_out

    # This method calculates the cost
    def calculate_cost(self, y, t):
        y = np.log(y)
        return np.sum(y*(t)) - self.l/2 * (np.linalg.norm(self.w0, 'fro')**2 + np.linalg.norm(self.w1, 'fro')**2)

    # This method calculates the approximations of the derivatives
    # and then compares them with what the neural network computed
    def gradCheck(self, xarg, targ):
        w0 = self.w0
        w1 = self.w1
        e = 1e-6
        samples = np.random.randint(xarg.shape[0], size=5)
        x = np.array(xarg[samples, :])
        t = np.array(targ[samples, :])

        w0t = self.w0

        cost, hidden_out, final_out = self.feed_forward(x, t)
        gradw0, gradw1 = self.computeGrads(x, hidden_out, final_out, t)

        numerical_approx = np.zeros(gradw0.shape) # stores all the numerical approximations
        print(gradw0.shape, gradw1.shape, w0.shape, w1.shape)
        for k in range(numerical_approx.shape[0]):
            for d in range(numerical_approx.shape[1]):
                w_tmp = np.copy(w0)
                w_tmp[k, d] += e
                self.w0 = w_tmp;
                costeplus, hidden_out, final_out = self.feed_forward(x, t)

                w_tmp = np.copy(w0)
                w_tmp[k, d] -= e
                self.w0 = w_tmp;
                costeminus, hidden_out, final_out = self.feed_forward(x, t)

                numerical_approx[k, d] = (costeplus - costeminus) / (2 * e)

        print("For W0, the maximum difference between the numerical gradient and the one we found is: ", np.max(np.abs(gradw0 - numerical_approx)))
        self.w0 = w0t
        numerical_approx = np.zeros(gradw1.shape)

        for k in range(numerical_approx.shape[0]):
            for d in range(numerical_approx.shape[1]):
                # Calculate W1 gradient
                w_tmp = np.copy(w1)
                w_tmp[k, d] += e
                self.w1 = w_tmp;
                costeplus, hidden_out, final_out = self.feed_forward(x, t)

                w_tmp = np.copy(w1)
                w_tmp[k, d] -= e

                self.w1 = w_tmp;
                costminus, hidden_out, final_out = self.feed_forward(x, t)

                numerical_approx[k, d] = (costeplus - costminus) / (2 * e)

        print("For W1, the maximum difference between the numerical gradient and the one we found is: ", np.max(np.abs(gradw1 - numerical_approx)))

    # This method tests the trained neural network
    def testNet(self):
        correct = 0
        # first we shuffle the test data
        self.test_data, self.test_1hv = self.shuffle_data(self.test_data, self.test_1hv)
        epochs = 1
        batch = 1 # we do batches of 1 data so that we can test each data separately
        finalCost = None
        for i in range(epochs):
            # first we shuffle the test data
            self.test_data, self.test_1hv = self.shuffle_data(self.test_data, self.test_1hv)
            if batch > len(self.test_data):
                print("ERROR: batch size is greater than the total number of training data")
                return None

            begin = 0
            end = batch

            while begin < end:
                data_batch = self.test_data[begin:end]
                data_batch = np.array([x for x in data_batch])
                hot_vec = np.array(self.test_1hv[begin:end])

                cost, hidden_out, final_out = self.feed_forward(data_batch, hot_vec)
                finalCost  = cost
                expected = np.argmax(hot_vec[0])
                predicted = np.argmax(final_out[0])
                #print(final_out[0])
                #print(expected,' ', predicted)
                if expected == predicted:
                    correct += 1

                begin = end
                end += batch
                if end > len(self.test_data):
                    end = len(self.test_data)

        return float(correct)/len(self.test_data), finalCost # return the accuraty and the cost of the final data

if __name__ == '__main__':

    funcs = [(func1, func1der), (func2, func2der), (func3, func3der)]

    epochs = 20

    hidden = [100, 200, 300]
    # change the data set paths
    dataSets = ['C:/Users/Alexandros/Downloads/mnistdata', 'C:/Users/Alexandros/Downloads/cifar-10-python/cifar-10-batches-py']
    # change the path of the output file
    with open('C:/Users/Alexandros/Desktop/results.txt', 'w') as f:
        for isMnist in [True, False]:
            if isMnist:
                for size in hidden:
                    for i in range(len(funcs)):
                        print('mnist ', size, ' ', i)
                        net = NeuNet(size, 10, funcs[i][0], funcs[i][1], batch=100, lm=0.01)
                        net.readMnistData(dataSets[0])
                        net.train_net(epochs)
                        accuracy, cost = net.testNet()

                        f.write('mnist(hidden = ' + str(size) + ', activation_function = ' + str(i) + ', batch = 100, lambda = 0.01, epochs = ' + str(epochs) + ', accuracy = ' + str(accuracy) + ', cost = ' + str(cost) +')\n')
            else:
                for size in hidden:
                    for i in range(len(funcs)):
                        print('cifar ', size, ' ', i)
                        net = NeuNet(size, 10, funcs[i][0], funcs[i][1], batch=100, lm=0.1)
                        net.readCIFAR10Data(dataSets[1])
                        net.train_net(epochs)
                        accuracy, cost = net.testNet()
                        f.write('cifar(hidden = ' + str(size) + ', activation_function = ' + str(
                        i) + ', batch = 100, lambda = 0.1, epochs = ' + str(epochs) + ', accuracy = ' + str(accuracy) + ', cost = ' + str(cost) +')\n')

