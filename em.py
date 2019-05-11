import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def read_image(path):
    img = np.array(mpimg.imread(path))
    print('original:', img.shape)
    height, width, _ = img.shape
    img = img.reshape(width*height, 3)
    img = img.astype(float) / 255
    return img, width, height

def mix(sk, x, mk):
    print(mk.shape)
    print(x.shape)
    return (1/np.sqrt(2*np.pi*sk))*np.exp(-(1/(2*sk))*((x - mk)**2))

def l(px):
    px = np.log(px)
    return np.sum(px, axis = 0)

def expectation_maximization(X, w, h, K):
    # the array that will hold the g values for each x
    gvals = np.zeros([X.shape[0],K]) + 0.00001
    sigma_squared = np.random.rand(K,1)
    mvals = np.random.rand(K,X.shape[1])
    pvals = np.full((K,1), 1/K)
    #print(X.shape)
    #print(gvals.shape)
    #print(sigma_squared.shape)
    #print(mvals.shape)
    #print(pvals.shape)
    i = 0
    gs_k_old = np.sum(gvals, axis = 1)
    while True:
        i += 1
        gvals_n = np.zeros([X.shape[0],K])
        sigma_squared_n = np.zeros([K,1])
        mvals_n = np.zeros([K,X.shape[1]])
        pvals_n = np.zeros([K,1])
        
        # expectation stage
        for k in range(K):
            gvals_n[:, k] = pvals[k]*np.prod((1/np.sqrt(2*np.pi*sigma_squared[k]))*\
                      np.exp(-(1/(2*sigma_squared[k]))*((X - mvals[k])**2)), axis = 1) # numpy allows NxD - 1XD = NxD

        gs_k = np.sum(gvals_n, axis = 1)
        #print('testssss', gs_k.shape)
        #print(gs)
        for k in range(K):
            gvals_n[:, k] = gvals_n[:, k] / gs_k

        # test the expectation stage
        #print(np.sum(gvals, axis = 1)[:5])
        gs_n = np.sum(gvals_n,  axis = 0)
        
        #print('ms shape:', gs_n.shape)
        # maximization stage
        for k in range(K):
            g = np.array(gvals_n[:, k], ndmin=2).T
            t = np.array(X*g, ndmin = 2)
            #print(t.shape)
            #print(ms[k].shape)
            mvals_n[k] = np.sum(t, axis = 0) / gs_n[k]

        #print(mvals)

        for k in range(K):
            temp_dif = X - mvals_n[k,:]
            temp_dif = np.power(temp_dif, 2)
            #print('temp_dif', temp_dif.shape)
            temp = np.array(np.sum(temp_dif, axis = 1), ndmin=2).T
            #print('temp', temp.shape)
            temp2 = np.array(gvals_n[:, k], ndmin=2)
            #print('temp2', temp2.shape)
            sigma_squared_n[k] = temp2.dot(temp)/(X.shape[1] * gs_n[k])
            #temp = temp *
        #print(sigma_squared)

        for k in range(K):
            pvals_n[k] = gs_n[k]/X.shape[0]
        
        i += 1
        lnew = l(gs_k)
        lold = l(gs_k_old)
        dif = lnew - lold
        print(dif)
        print('lnew =', lnew,'\nlold =', lold)

        pvals = np.copy(pvals_n)
        gvals = np.copy(gvals_n)
        mvals = np.copy(mvals_n)
        sigma_squared = np.copy(sigma_squared_n)
        gs_k_old = gs_k
        if dif < 0:
            print('Error occured!\nIterations: ', i)
            break
        if dif < 0.1:
            print('Converged!\niterations', i)
            kapas = np.argmax(gvals, axis = 1)
            new_img = np.array(list(map(lambda k: mvals[k], kapas)))
            temp = new_img
            new_img = new_img.reshape(h,w,3)
            print('new image:', new_img.shape)
            print(h,w)
            imgplot = plt.imshow(new_img)
            #plt.show()
            error = np.sum(np.power(np.linalg.norm(X - temp),2))/X.shape[0]
            print('error:',error) 
            plt.savefig(str(K) + 'im.jpg')

            break

                
if __name__ == '__main__':
    np.random.seed(1993)
    img, w, h = read_image('im2.jpg')
    for k in [6]:
        print('K =',k)
        expectation_maximization(img, w, h, k)
