from binascii import b2a_base64
from code import interact
from sqlite3 import dbapi2
from tkinter import Y
from xml.etree.ElementPath import prepare_predicate
import numpy as np # import numpy, simplifies matrixes and linear algebra A LOT  //Problem för mig med denna rad.
import math # imports math library, holds a lot of essential math 
# This file will take data from our dataset and train an algorithm by K-nearest-neighbors algorithm
# [ 1 2 3 4
#   1 2 3 4
#   1 2 3 4]

data = np.array(data)
m, n =data.shape
np.random.shuffle(data)

data_dev = data[0:1000]
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].t
Y_train = data_train[0]
X_train = data_train[1:n]

def init_prarams():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forword_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1) 
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A1)

def one_hot():
    one_hot_Y = np.zero((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = np.zero((Y.size, Y.max() + 1))
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X. T)
    db1 =  1 / m * np.sum(dZ2, 2)
    return dW1, db1, dW2, db2

def update_params(dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accurcy(prepare_predicate, Y):
    print(prepare_predicate, Y)
    return np.sum(prepare_predicate == Y) / Y.size

def gradient_decent(X, Y, iterations, alpha):
    for i in range (iterations):
        Z1, A1, Z2, A2 = forword_prop(W1, b1 ,W2, b2)
        dW1, db1, dW2, db2 =back_prop(Z1, A1, Z2, A2, X, Y)
        W1,b1,W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteratiom",i)
            print("Accurcy", get_accurcy(get_prediction(A2),Y))
            if(i % 10 == 0):
                print("interation", i)
                print("accurcy", get_accurcy(get_prediction(A2),Y))
        return W1, b1, W2, b2
 
W1, b1, W2, b2 = gradient_decent(X_train, Y_train, 500, 0.1)


#25:11

