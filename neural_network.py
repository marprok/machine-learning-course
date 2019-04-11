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
        self.test = None
        self.inlen = 0
        self.w0 = None
        self.batch = batch

    def shuffle_data(self):
        # zip the one hot vector and the training data together
        temp = list(zip(self.train_data, self.train_1hv))
        # shuffle them
        shuffle(temp)
        # unzip
        self.train_data, self.train_1hv = zip(*temp)
    
    def train_net(self, epochs):
        for i in range(epochs):

            # first we shuffle the training data
            self.shuffle_data()
            cost = self.feed_forward()
            print('cost: ',cost,' epoch = ', i)

            
        
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


    def compute_gradients_cost(self, data_batch, hidden_out, batch_dot_w0, hidden_dot_w1, output, onehotvec,lamda = 0):
        #Z = get_z(X,w1) # hidden_out

        # The result of Z*w2
        #z_w2 = Z.dot(w2.T) # hidden_dot_w1

        #Y = softmax(z_w2) #output
        # Compute the cost function to check convergence
        max_error = np.max(hidden_dot_w1, axis=1)
        Ew = np.sum(onehotvec * hidden_dot_w1) - np.sum(max_error) - \
             np.sum(np.log(np.sum(np.exp(hidden_dot_w1 - np.array([max_error, ] * hidden_dot_w1.shape[1]).T), 1))) - \
             (0.5 * lamda) * (np.sum(np.square(self.w0)) + np.sum(np.square(self.w1)))
        
        # Calculate gradient for w1
        grad_w1 = (onehotvec - output).T.dot(hidden_out) - lamda * self.w1
        
        # We remove the bias since z0 is not dependant by w1
        w1_temp = np.copy(self.w1[:, 1:])
        
        # This is the result of the derivative of the activation function
        der = self.activation_der(data_batch.dot(self.w0.T))
        
        temp = (onehotvec - output).dot(w1_temp) * der
        
        # Calculate gradient for w0
        grad_w0 = temp.T.dot(data_batch) - lamda*self.w0
        
        
        return Ew, grad_w0, grad_w1


    
    def feed_forward(self):

        if self.batch > len(self.train_data):
            print("ERROR: batch size is greater than the total number of training data")
            return None
        
        begin, end = 0, self.batch
        cost = None
        while begin < end:
            #print('begin: ' + str(begin) +'\nend: ' + str(end))
            data_batch = self.train_data[begin:end]
            data_batch = np.array([x for x in data_batch])
            temp0 = data_batch.dot(self.w0.T)
            hidden_leyer_out = self.activation(temp0)
            hidden_leyer_out = np.hstack((np.ones((hidden_leyer_out.shape[0], 1)), hidden_leyer_out))  # +1 col at the start of the array for the bias term
            
            temp1 = hidden_leyer_out.dot(self.w1.T)
            final_out = softmax(temp1)
            hot_vec = np.array(self.train_1hv[begin:end])

            #print('cost: ',cost)

            error, dw0, dw1 = self.compute_gradients_cost(data_batch, hidden_leyer_out, temp0, temp1, final_out, hot_vec,lamda = 0.01)
            self.w0 += 0.005*dw0
            self.w1 += 0.005*dw1
            #print('error: ',error)
            cost = self.calculate_cost(final_out, hot_vec)
            #print('cost: ', cost)
            '''for current in range(begin, end):
                input_vec = np.array(self.train_data[current], ndmin=2)
                
                #input_vec = data
                #input_vec = np.hstack((input_vec, np.array([1], ndmin=2)))# +1 for the bias term
                #print(input_vec.shape)
                #print(self.w0.shape)
                temp = input_vec.dot(self.w0.transpose())
                #print(temp.shape)
                #hidden_layer_out = np.array(list(map(self.activation,temp)),ndmin=2)
                hidden_leyer_out = []
                #print(temp)
                #for i in range(temp.shape[1]):
                #    hidden_leyer_out.append(self.activation(temp[0][i]))
                hidden_leyer_out = self.activation(temp)
                hidden_leyer_out = np.array(hidden_leyer_out, ndmin=2)
                hidden_leyer_out = np.hstack((hidden_leyer_out, np.array([1], ndmin=2)))# +1 for the bias term
                #print(hidden_leyer_out.shape)
                temp = hidden_leyer_out.dot(self.w1.transpose())
                #print(temp)
                #print(temp.shape)
                # now temp contains the dot products of the second half
                # and the only thing that is left is to give it as
                # input to the softmax function
                final_out = softmax(temp)
                #print(final_out)
                print('cost: ',self.calculate_cost(final_out, self.train_1hv[current]))
                '''
            begin = end
            end += self.batch
            if end > len(self.train_data):
                end = len(self.train_data)
             
        '''for data in self.train_data: # for each training data
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
            print('cost: ',self.calculate_cost(final_out, self.train_1hv[j]))
            if j == 5:
                break
            j += 1'''
        return cost

    def calculate_cost(self, y, t):
        #print(y)
        y = np.log(y)
        #print(y)
        l = 0.01
        return np.sum(y*(t)) - l/2 * (np.linalg.norm(self.w0, 'fro')**2 + np.linalg.norm(self.w1, 'fro')**2)

    def gradcheck_softmax(self, lamda):


        epsilon = 1e-6

        _list = np.random.randint(self.train_data.shape[0], size=5)
        x_sample = np.array(self.train_data.shape[0][_list, :])
        t_sample = np.array(self.train_1hv.shape[0][_list, :])
        # split the training for each batch into two methods 1 feed_forward and one back_propagate for a single epoch
        # then feed forward the x_sample and compute the gradients wihtout updating the weights
        Ew, gradW0, gradW1 = self.compute_gradients_cost(data_batch, hidden_leyer_out, temp0, temp1, final_out, hot_vec,lamda = 0.01)
        # then create two new matrixes w0',w1'

        cost = self.calculate_cost()
        print("gradEw shape: ", gradEw.shape)

        numericalGrad = np.zeros(gradEw.shape)
        # Compute all numerical gradient estimates and store them in
        # the matrix numericalGrad

        # loop for w0
        for k in range(numericalGrad.shape[0]):
            for d in range(numericalGrad.shape[1]):
                # add epsilon to the w[k,d]
                w_tmp = np.copy(W)
                w_tmp[k, d] += epsilon
                e_plus, _ = cost_grad_softmax(w_tmp, x_sample, t_sample, lamda)

                # subtract epsilon to the w[k,d]
                w_tmp = np.copy(W)
                w_tmp[k, d] -= epsilon
                e_minus, _ = cost_grad_softmax(w_tmp, x_sample, t_sample, lamda)

                # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
                numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)
        # loop for w1
        for k in range(numericalGrad.shape[0]): # loop for each element
            for d in range(numericalGrad.shape[1]):
                # add epsilon to the w[k,d]
                w_tmp = np.copy(W)
                w_tmp[k, d] += epsilon
                #feed forward and calculate the cost without updating
                e_plus, _ = cost_grad_softmax(w_tmp, x_sample, t_sample, lamda)

                # subtract epsilon to the w[k,d]
                w_tmp = np.copy(W) # reset the W matrix to it's original state
                w_tmp[k, d] -= epsilon
                e_minus, _ = cost_grad_softmax(w_tmp, x_sample, t_sample, lamda)

                # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
                numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)


        return (gradEw, numericalGrad)


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
    net.readMnistData('/home/p3150141/Downloads/')
    net.train_net(100)
