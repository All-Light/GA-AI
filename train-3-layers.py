import numpy as np # simplifies matrixes and linear algebra 
from datareader import MnistDataloader # See datareader.py 
#import math # imports math library, holds a lot of essential math 
import matplotlib.pyplot as plt # for plotting graphs

# TODO credit original video/guide for both training and mnist dataloader!!!!!!!!
# 


# hardcoded paths to training/testing data
training_images_filepath = './dataset/training/train-images-idx3-ubyte'
training_labels_filepath = './dataset/training/train-labels-idx1-ubyte'
test_images_filepath = './dataset/testing/t10k-images-idx3-ubyte'
test_labels_filepath = './dataset/testing/t10k-labels-idx1-ubyte'

# see datareader.py
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(X_train, Y_train), (X_test, Y_test) = mnist_dataloader.load_data()

# training data used to train the model
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# testing data used to test the model after training
X_test = np.array(X_test)
Y_test = np.array(Y_test)

SCALE_FACTOR = 255
WIDTH = X_train.shape[1]
HEIGHT = X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR

# The network has 3 layers (input layer, hidden layer, output layer)

# W1 = Weights between nodes in input layer and hidden layer
# W2 = Weights between nodes in hidden layer and output layer
# b1 = Biases for nodes in input layer
# b2 = Biases for nodes in hidden layer
# 
# X = values for nodes in input layer (image data)
# Z1 = Values in hidden layer
# A1 = normalized Z1 (between 0 and 1) 
# Z2 = Values in output layer
# A2 = normalized Z2 (between 0 and 1) 
#


def init_params(size):
    # random initalization 
    W1 = np.random.rand(10, size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3  = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    # exp is the array Z where all values are between 0-1 then put as exponents to "e" e.g e^0.1, e^0.2525324, ...
    exp = np.exp(Z - np.max(Z)) 
    return exp / exp.sum(axis=0)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    # this function will run the neural network with the current weights and biases (with X as input image)
    # A2 is the output layer
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m):
    one_hot_Y = one_hot(Y)

    dZ3 = 2*(A3 - one_hot_Y) #10, m
    dW3 = 1/m * (dZ3.dot(A2.T)) # 10 , 10
    db3 = 1/m * np.sum(dZ3,1) # 10, 1

    #dZ2 = W3.T.dot(dZ3)*deriv_ReLU(Z2) # 10, m
    #dW2 = 1/m * (dZ2.dot(X.T)) #10, 784
    #db2 = 1/m * np.sum(dZ2,1) # 10, 1

    dZ2 = 2*(A2 - one_hot_Y) #10, m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1

    dZ1 = W2.T.dot(dZ2)*deriv_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    # changes weights and biases with step 
    '''
    print(dW1.size)
    print(dW2.size)
    print(dW3.size)

    print(db1.size)
    print(db2.size)
    print(db3.size)
    '''
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))
    W3 -= alpha * dW3
    b3 -= alpha * np.reshape(db3, (10,1))

    return W1, b1, W2, b2, W3, b3

def get_prediction(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    # get every correct prediction divided by total amount of images
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1 ,b1, W2, b2, W3, b3):
    # only get A2 (output layer), aka the prediction
    _, _, _, _, _, A3 = forward_prop(W1, b1 , W2, b2, W3, b3, X)
    predictions = get_prediction(A3)
    return predictions

def show_prediction(index, X, Y, W1, b1, W2, b2, W3, b3):
    # show the prediction and image in pyplot
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# Main function. 
def gradient_decent(X, Y, iterations, alpha, test_x, test_y):
    size, m = X.shape
    # initialize parameters (randomized)
    W1, b1, W2, b2, W3, b3 = init_params(size)
    # we run propogation a number of times (iterations)
    for i in range(iterations):
        # make prediction
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1 , W2, b2, W3, b3, X)

        # get change in weights and biases from prediction
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m)
        
        # update weights and biases from prediction
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 50 == 0:
            print("Iteration",i)
            print("Accuracy", get_accuracy(get_prediction(A3),Y))
    print("New values. Iterations:", iterations, " alpha: ", alpha)
    
    # This displays the ACTUAL accuracy of the model, previous accuracies are merely an approximation
    # this is because they are testing the model on the training data, whereas this line uses the testing data
    print("Accuracy: ", get_accuracy(make_predictions(test_x, W1 ,b1, W2, b2, W3, b3),test_y))
    preds.append(("Steps: ", iterations, " alpha: ", alpha, " Accuracy: ", get_accuracy(make_predictions(test_x, W1 ,b1, W2, b2, W3, b3),test_y)))
    return W1, b1, W2, b2, W3, b3

# training with different iterations and alpha
preds =  []
for i in range(1,6):
    W1, b1, W2, b2, W3, b3 = gradient_decent(X_train, Y_train, 30000, 0.1*i, X_test, Y_test)
    print(preds)
