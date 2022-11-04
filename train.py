#from binascii import b2a_base64
#from code import interact
#from sqlite3 import dbapi2
#from tkinter import Y
#from xml.etree.ElementPath import prepare_predicate
import numpy as np # import numpy, simplifies matrixes and linear algebra A LOT  //Problem för mig med denna rad.
from datareader import MnistDataloader
import math # imports math library, holds a lot of essential math 
# This file will take data from our dataset and train an algorithm by K-nearest-neighbors algorithm
# [ 1 2 3 4
#   1 2 3 4
#   1 2 3 4]


# NOTE this file cant be run yet!


training_images_filepath = './dataset/training/train-images-idx3-ubyte'
training_labels_filepath = './dataset/training/train-labels-idx1-ubyte'
test_images_filepath = './dataset/testing/t10k-images-idx3-ubyte'
test_labels_filepath = './dataset/testing/t10k-labels-idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
train, test = mnist_dataloader.load_data()


data = np.array(train, dtype=object)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_train = data_dev[0]
X_train = data_dev[1:n]
#Y_dev = data_dev[0]
#X_dev = data_dev[1:n]

#data_train = data[1000:m].T
#Y_train = data_train[0]
#X_train = data_train[1:n]


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot():
    one_hot_Y = np.zero((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
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
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1 , W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteratiom",i)
            print("Accurcy", get_accurcy(get_prediction(A2),Y))
            if(i % 10 == 0):
                print("interation", i)
                print("accurcy", get_accurcy(get_prediction(A2),Y))
        return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_decent(X_train, Y_train, 500, 0.1)


#25:11

