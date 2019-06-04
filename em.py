import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# This function reads the image, reshapes it into a 2D array and then
# returns the normalized buffer along side with the width and height.
def read_image(path):
    img = np.array(mpimg.imread(path))
    print('original:', img.shape)
    height, width, _ = img.shape
    img = img.reshape(width*height, 3)
    img = img.astype(float) / 255
    return img, width, height

# This function calculates the logarithmic probability
def l(px):
    px = np.log(px)
    return np.sum(px, axis = 0)

# This function is the actual implementation of the EM algorithm.
# X = data input, w = image width, h = image height, K = the K value
def expectation_maximization(X, w, h, K):
    # Here we initialize the arrays that we are going to use.
    gvals = np.zeros([X.shape[0],K]) + 0.00001 # the array that will hold the g values for each x
    sigma_squared = np.random.uniform(0.4, 0.8, (K,1)) # the array that will hold the sigma(squared) values 
    mvals = np.random.uniform(0, 1, (K,X.shape[1])) # the array that will hold the m values
    pvals = np.full((K,1), 1/K) # the array that will hold the p values
    i = 0
    gs_k_old = np.sum(gvals, axis = 1) # store the sum of the g values per row
    
    while True:
        i += 1
        # these are the arrays for the new iteration
        gvals_n = np.zeros([X.shape[0],K])
        sigma_squared_n = np.zeros([K,1])
        mvals_n = np.zeros([K,X.shape[1]])
        pvals_n = np.zeros([K,1])
        
        # expectation stage
        # first calculate the numerator of g and we add a very small value to the sigma
        # in order to prevent the division by zero
        for k in range(K):
            gvals_n[:, k] = pvals[k]*np.prod((1/(np.sqrt(2*np.pi*sigma_squared[k]) + 1e-100))*\
                      np.exp(-(1/(2*sigma_squared[k] + 1e-100))*((X - mvals[k])**2)), axis = 1) # numpy allows NxD - 1XD = NxD

        # then we calculate the denominator by summing each row of g(for each k)
        gs_k = np.sum(gvals_n, axis = 1)
        for k in range(K):
            gvals_n[:, k] = gvals_n[:, k] / gs_k

        # calculate the sum of g per column(for a specific k)
        gs_n = np.sum(gvals_n,  axis = 0)
        
        # maximization stage
        # calculate the values of m
        for k in range(K):
            # numpy wouldn't allow to do cumputations without explicitly stating the dimentions
            g = np.array(gvals_n[:, k], ndmin=2).T # for a specific k, take the values for each data
            t = np.array(X*g, ndmin = 2)
            mvals_n[k] = np.sum(t, axis = 0) / gs_n[k]

        # calculate the sigma squared values
        for k in range(K):
            temp_dif = X - mvals_n[k,:]
            temp_dif = np.power(temp_dif, 2)
            # sum the difference per row(for all the D values)
            temp = np.array(np.sum(temp_dif, axis = 1), ndmin=2).T
            # again I had to explicitly state the dimentions
            temp2 = np.array(gvals_n[:, k], ndmin=2)
            sigma_squared_n[k] = temp2.dot(temp)/(X.shape[1] * gs_n[k])

        # calculate the values of p
        for k in range(K):
            pvals_n[k] = gs_n[k]/X.shape[0]

        # calculate the new logarithmic probability
        lnew = l(gs_k)
        # calculate the old logarithmic probability
        lold = l(gs_k_old)
        
        dif = lnew - lold
        print(dif)
        print('lnew =', lnew,'\nlold =', lold)
        # update the arrays
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
            imgplot = plt.imshow(new_img)
            error = np.sum(np.power(np.linalg.norm(X - temp),2))/X.shape[0]
            print('error:',error) 
            plt.savefig(str(K) + 'im.jpg')
            break

                
if __name__ == '__main__':
    #np.random.seed(123123)
    img, w, h = read_image('im.jpg')
    for k in [1,2,4,8]:
        print('K =',k)
        expectation_maximization(img, w, h, k)
