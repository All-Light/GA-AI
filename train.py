import numpy as np # import numpy, simplifies matrixes and linear algebra A LOT  //Problem för mig med denna rad.
from datareader import MnistDataloader
import math # imports math library, holds a lot of essential math 
import matplotlib.pyplot as plt # for plotting graphs

training_images_filepath = './dataset/training/train-images-idx3-ubyte'
training_labels_filepath = './dataset/training/train-labels-idx1-ubyte'
test_images_filepath = './dataset/testing/t10k-images-idx3-ubyte'
test_labels_filepath = './dataset/testing/t10k-labels-idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(X_train, Y_train), (X_test, Y_test) = mnist_dataloader.load_data()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

SCALE_FACTOR = 255
WIDTH = X_train.shape[1]
HEIGHT = X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR

def init_params(size):
    W1 = np.random.rand(10, size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) 
    return exp / exp.sum(axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*deriv_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))
    return W1, b1, W2, b2

def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1 , W2, b2, X)
    predictions = get_prediction(A2)
    return predictions

def show_prediction(index, X, Y, W1, b1, W2, b2):
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def gradient_decent(X, Y, iterations, alpha, test_x, test_y):
    size, m = X.shape
    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1 , W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 1000 == 0:
            print("Iteration",i)
            print("Accuracy", get_accuracy(get_prediction(A2),Y))
    print("New values. Iterations:", iterations, " alpha: ", alpha)
    print("Accuracy: ", get_accuracy(make_predictions(test_x, W1 ,b1, W2, b2),test_y))
    preds.append(("Steps: ", iterations, " alpha: ", alpha, " Accuracy: ", get_accuracy(make_predictions(test_x, W1 ,b1, W2, b2),test_y)))
    return W1, b1, W2, b2


preds =  []
for i in range(1,6):
    W1, b1, W2, b2 = gradient_decent(X_train, Y_train, 30000, 0.1*i, X_test, Y_test)
    print(preds)
