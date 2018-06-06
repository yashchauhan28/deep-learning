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

Y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10009):

    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    error = Y - l1

    l1delta = error * nonlin(l1,True)

    syn0 += np.dot(l0.T,l1delta)

print ("Output after training")
print (l1)
