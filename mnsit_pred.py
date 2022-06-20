from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./data/digit-recognizer/train.csv')
data.head()

# get the data from aframe to a np array
data = np.array(data)
# m = amount of data we have(x)
# n = numof features + 1 because of label col
m, n = data.shape

# shuffle before splitting
np.random.shuffle(data)

# first 1000 examples / vals
dataRTest = data[0:1000].T

yRTest = dataRTest[0]
xRTest = dataRTest[1:n]
xRTest = xRTest / 255.0

# rest of data is train
dataTrain = data[1000:m].T
yTrain = dataTrain[0]
xTrain = dataTrain[1:n]
xTrain = xTrain / 255.0
_,mTrain = xTrain.shape

print(yTrain)

def initParams():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def fwdProp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def reLUDeriv(Z):
    return Z > 0

def oneHotEncode(Y):
    oneHotEncodeY = np.zeros((Y.size, Y.max() + 1))
    oneHotEncodeY[np.arange(Y.size), Y] = 1
    oneHotEncodeY = oneHotEncodeY.T
    return oneHotEncodeY

def bwdProp(Z1, A1, Z2, A2, W1, W2, X, Y):
    oneHotEncodeY = oneHotEncode(Y)
    dZ2 = A2 - oneHotEncodeY
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * reLUDeriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def GD(X, Y, alpha, iterations):
    print("Running GD!")
    W1, b1, W2, b2 = initParams()
    for i in range(iterations):
        Z1, A1, Z2, A2 = fwdProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = bwdProp(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = getPredictions(A2)
            print(getAccuracy(predictions, Y))
    return W1, b1, W2, b2




def makePrediction(X, W1, b1, W2, b2):
    _, _, _, A2 = fwdProp(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions

def testPrediction(index, W1, b1, W2, b2):
    current_image = xTrain[:, index, None]
    prediction = makePrediction(xTrain[:, index, None], W1, b1, W2, b2)
    label = yTrain[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    

W1, b1, W2, b2 = GD(xTrain, yTrain, 0.50, 500)

testPrediction(0, W1, b1, W2, b2)
testPrediction(1, W1, b1, W2, b2)
testPrediction(2, W1, b1, W2, b2)
testPrediction(3, W1, b1, W2, b2)