import matplotlib
import matplotlib.image as mpimg
import numpy as np

def read_image(path):
    img = np.array(mpimg.imread(path))
    width, height, _ = img.shape
    img = img.reshape(width*height, 3)
    img = img.astype(float) / 255
    return img

def mix(sk, x, mk):
    print(mk.shape)
    print(x.shape)
    return (1/np.sqrt(2*np.pi*sk))*np.exp(-(1/(2*sk))*((x - mk)**2))
    
def expectation_maximization(X, K):
    # the array that will hold the g values for each x
    gvals = np.zeros([X.shape[0],K])
    sigma_squared = np.random.rand(K,1)
    mvals = np.random.rand(K,X.shape[1])
    pvals = np.full((K,1), 1/K)
    #print(X.shape)
    #print(gvals.shape)
    #print(sigma_squared.shape)
    #print(mvals.shape)
    #print(pvals.shape)
    i = 0
    while i < 10:
        # expectation stage
        for k in range(K):
            gvals[:, k] = pvals[k]*np.prod((1/np.sqrt(2*np.pi*sigma_squared[k]))*\
                      np.exp(-(1/(2*sigma_squared[k]))*((X - mvals[k])**2)), axis = 1) # numpy allows NxD - 1XD = NxD

        gs = np.sum(gvals, axis = 1)
        #print(gs)
        for k in range(K):
            gvals[:, k] = gvals[:, k] / gs

        # test the expectation stage
        #print(np.sum(gvals, axis = 1)[:5])
        gs = np.sum(gvals,  axis = 0)
        print('ms shape:', gs.shape)
        # maximization stage
        for k in range(K):
            g = np.array(gvals[:, k], ndmin=2).T
            t = np.array(X*g, ndmin = 2)
            #print(t.shape)
            #print(ms[k].shape)
            mvals[k] = np.sum(t, axis = 0) / gs[k]

        #print(mvals)

        for k in range(K):
            temp_dif = X - mvals[k,:]
            temp_dif = np.power(temp_dif, 2)
            print('temp_dif', temp_dif.shape)
            temp = np.array(np.sum(temp_dif, axis = 1), ndmin=2).T
            print('temp', temp.shape)
            temp2 = np.array(gvals[:, k], ndmin=2)
            print('temp2', temp2.shape)
            sigma_squared[k] = temp2.dot(temp)/(X.shape[1] * gs[k])
            #temp = temp *
        #print(sigma_squared)

        for k in range(K):
            pvals[k] = gs[k]/X.shape[0]
        
        i += 1

    
if __name__ == '__main__':
    img = read_image('im.jpg')
    expectation_maximization(img, 2)
