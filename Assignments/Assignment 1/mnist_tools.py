"""
Tools for loading the MNIST Data.

@author: Brett
"""

import scipy.io
import numpy as np

_mfile = None
train = []
test = []

def _load_data(filename) :
    global _mfile
    if _mfile is None :
        _mfile = scipy.io.loadmat(filename)

def _load_digit_data(digit,typename) :
    global _mfile
    data = _mfile[typename+str(digit)]
    return data

"""
(Internal function not used by students.)
Returns a tuple (train,test).  Each element of the tuple is a 10-element list.
train[i] and test[i] correspond to images of the digit i.  train[i] and test[i]
are themselves 2d numpy arrays.  Each row of the array has 784 columns
corresponding to a 28x28 image of a handwritten digit.  filename contains 
the name of the file containing the MNIST data (as a Matlab .mat file).
"""
def _load_mnist_data(filename) :
    global train, test
    _load_data(filename)
    np.random.seed(777)
    if len(train) == 0 :
        for i in range(10) :
            train.append(_load_digit_data(i,"train"))
            np.random.shuffle(train[i])
            test.append(_load_digit_data(i,"test"))
            np.random.shuffle(test[i])
    return (train,test)

"""
Returns a 10-element list train of training data.
train[i] corresponds to images of the digit i.  train[i]
is a 2d numpy array.  Each row of the array has 784 columns
corresponding to a 28x28 image of a handwritten digit.  
filename contains the name of the file containing the 
MNIST data (as a Matlab .mat file).
"""
def load_train_data(filename) :
    (train,test) = _load_mnist_data(filename)
    #Restrict to 100 samples from each digit
    for i in range(10) :
        train[i] = train[i][0:100,:]
    return train

"""
Returns a tuple (test,testLabels).  test is a 2d numpy array of shape (5,784)
where each row is a test image.  labels is a list of length 5 containing 
the digit for each row of test.
"""
def load_test_data(filename) :
    (train,test) = _load_mnist_data(filename)
    np.random.seed(200)
    ret = []
    testLabels = []
    for i in range(5) :
        dig = np.random.randint(0,len(test))
        testLabels.append(dig)
        digTests = test[dig]
        img = digTests[np.random.randint(0,digTests.shape[0]),:]
        ret.append(img)
    return (np.stack(ret),testLabels)

"""
Returns a tuple (test,noisyTest,testLabels).  
test and testLabels come from load_test_data.
noisyTest has the same shape as test, but has been
corrupted with Gaussian noise. 
"""
def load_noisy_test_data(filename) :
    test,testLabels = load_test_data(filename)
    np.random.seed(333)
    noisyTest = test + np.random.randn(test.shape[0],28*28)*255
    return (test,noisyTest,testLabels)
