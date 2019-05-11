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

    for k in range(K):
        gvals[:, k] = pvals[k]*np.prod((1/np.sqrt(2*np.pi*sigma_squared[k]))*np.exp(-(1/(2*sigma_squared[k]))*((X - mvals[k])**2)), axis = 1) # numpy allows NxD - 1XD = NxD
        print('test')
    gs = np.sum(gvals, axis = 1)
    for k in range(K):
        gvals[:, k] = gvals[:, k] / gs
    
    print(np.sum(gvals, axis = 1)[:5])



    
    '''for n in range(X.shape[0]):
        temp = []
        for k in range(K):
            temp.append(pvals[k]*mix(sigma_squared[k], X[n], mvals[k]).prod(axis=0))
        temp = np.array(temp)
        #print(temp.shape)
        s = np.sum(temp)
        for k in range(K):
            gvals[n][k] = temp[k]/s

        #print(temp/np.sum(temp))
        #print(temp.shape)
        temp.reshape(K)
        #print(temp.shape)
        #gvals[n] = temp/np.sum(temp)
        #print(gvals[n])
        break'''
    
if __name__ == '__main__':
    img = read_image('im.jpg')
    expectation_maximization(img, 2)
