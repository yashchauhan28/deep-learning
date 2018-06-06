import numpy as np

def nonlin(x,deri=False):
    if deri == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

Y = np.array([[0],[0],[1],[1]])

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for iter in range(100009):

    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = Y - l2

    if iter%10000 == 0:
        print ("Error : " , str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin(l2,True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1,True)

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)

print ("output is : " , l2)
