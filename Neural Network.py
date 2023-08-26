import random

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



house = pd.read_csv('datasets/hw3_house_votes_84.csv')
wine = pd.read_csv('datasets/hw3_wine.csv', sep='\t')
cancer = pd.read_csv('datasets/hw3_cancer.csv', sep='\t')



class Layer:

    #input_size = num of incoming neurons
    #output_size = num of output neurons aka output_size

    def __init__(self, input_size, num_neurons, activation, activation_prime):
        self.weights = np.random.randn(num_neurons, input_size) * 0.5
        self.biases = np.random.rand(1, num_neurons)[0]
        self.activation = activation
        self.activation_prime = activation_prime
        self.weight_gradients = []
        self.bias_gradients = []

    def forward_prop(self, input):
        self.input = input
        self.output = self.activation(np.dot(self.weights, self.input) + self.biases)
        return self.output

    def backward_prop(self, delta_next, alpha):

        delta = np.array([x * (1 - x) for x in self.input])*(delta_next @ self.weights)

        gradient = np.array(delta_next).T @ np.array([self.input])
        gradient_bias = delta_next[0]
        # print("Gradient of Weights: ")
        # print(gradient)
        # print("Gradient of Bias: ")
        # print(gradient_bias)

        self.weight_gradients.append(gradient)
        self.bias_gradients.append(gradient_bias)
        return delta

    def backward_prop_reg(self, bias_gradients, weights_gradients, alpha, regularization):

        #so we need to take the gradients after proccessing every training instance
        #n = the number of training instances
        #P = lambda * self.weights
        #gradient = (sum of all gradients (we use vector addition here) + P)/n
        bias = np.sum(bias_gradients, axis = 0)/len(bias_gradients)
        weights = (np.sum(weights_gradients, axis = 0) + self.weights*regularization)/len(weights_gradients)
        # print("Bias Regularized Gradients")
        # print(bias)
        # print("Weights Regularized Gradients")
        # print(weights)
        self.weights -= alpha * weights
        self.biases -= alpha * bias


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_prime(x):
    return x*(1-x)

def oneHotEncode(y_value, arr_classes):

    index = np.where(arr_classes == y_value)[0]
    res = np.zeros(len(arr_classes))
    res[index] = 1
    return res

class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit_regularization(self, x_train, y_train, epochs, alpha, regularization):

        for i in range(epochs):
            num_samples = len(x_train)
            for row in range(num_samples):
                output = x_train[row]

                for layer in self.layers:
                    output = layer.forward_prop(output)

                delta = [output - y_train[row]]
                #print("Training Instance: ", row + 1)

                for layer in reversed(self.layers):
                    #print("Delta: ", delta)

                    delta = layer.backward_prop(delta, 1)

                #print("--------------------------------------")

            for layer in self.layers:
                layer.backward_prop_reg(layer.bias_gradients, layer.weight_gradients, alpha, regularization)


    def fit_schocastic(self, x_train, y_train, alpha):

        num_samples = len(x_train)

        for row in range(num_samples):
            output = x_train[row]

            for layer in self.layers:
                output = layer.forward_prop(output)

            delta = [output - y_train[row]]

            #print("Training Instance: ", row + 1)

            for layer in reversed(self.layers):
             #   print("Delta: ", delta)
                delta = layer.backward_prop(delta, alpha)


            #print("--------------------------------------")


    def predict(self, row):
        output = row
        for layer in self.layers:
            output = layer.forward_prop(output)
        return (output == output.max(keepdims=1)).astype(float)


def q6(X, Y):

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2)

    arr_classes = np.unique(y_train)
    y_train = [oneHotEncode(y, arr_classes) for y in y_train]
    arr_classes = np.unique(y_test)
    y_test = [oneHotEncode(y, arr_classes) for y in y_test]

    network = NeuralNetwork()
    layer1 = Layer(13, 8, sigmoid, sigmoid_prime)
    layer2 = Layer(8, 8, sigmoid, sigmoid_prime)
    layer3 = Layer(8, 3, sigmoid, sigmoid_prime)

    network.add(layer1)
    network.add(layer2)
    network.add(layer3)


    x_axis = []
    y_axis = []
    for i in range(1, len(x_train), 5):
        x_temp = x_train[i:i+5, :]
        y_temp = y_train[i:i+5]
        network.fit_regularization(x_temp, y_temp, 500, 0.5, 0.25)
        prediction = []

        for row in x_test:
            prediction.append(network.predict(row))

        x_axis.append(i)
        y_axis.append(accuracy_score(prediction, y_test))


    plt.plot(x_axis, y_axis)
    plt.show()

def strat_cross_validate(X, Y, splits):

    skf = StratifiedKFold(n_splits = splits, shuffle=True, random_state=1)
    total = 0

    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        arr_classes = np.unique(y_train)
        y_train = [oneHotEncode(y, arr_classes) for y in y_train]
        arr_classes = np.unique(y_test)
        y_test = [oneHotEncode(y, arr_classes) for y in y_test]

        network = NeuralNetwork()
        layer1 = Layer(16, 8, sigmoid, sigmoid_prime)
        layer2 = Layer(8, 8, sigmoid, sigmoid_prime)
        layer3 = Layer(8, 2, sigmoid, sigmoid_prime)
        #layer4 = Layer(6, 2, sigmoid, sigmoid_prime)

        network.add(layer1)
        network.add(layer2)
        network.add(layer3)
       # network.add(layer4)

        network.fit_regularization(x_train, y_train, 500, 0.5, 0.25)

        prediction = []
        f1scores = []
        accuracy = []
        for row in x_test:
            prediction.append(network.predict(row))

        f1scores.append(f1_score(prediction, y_test, average = 'macro'))
        accuracy.append(accuracy_score(prediction, y_test))


    print("Accuracy: ", sum(accuracy)/len(accuracy))
    print("F1: ", sum(f1scores)/len(f1scores))

def wineSet():

    wine = pd.read_csv('datasets/hw3_wine.csv', sep='\t')

    X = wine.iloc[:, 1:]
    Y = wine.iloc[:, 0].values.reshape(-1, 1)

    X = (X - X.min()) / (X.max() - X.min())
    X = X.values
    q6(X, Y)
    #strat_cross_validate(X,Y, 5)


#wineSet()

def houseSet():

    house_votes = pd.read_csv('datasets/hw3_house_votes_84.csv')

    X = house_votes.iloc[:, :-1]
    Y = house_votes.iloc[:, -1].values.reshape(-1, 1)

    X = (X - X.min()) / (X.max() - X.min())
    X = X.values

    q6(X, Y)

#houseSet()
def backpropChecker1():
    Layer1 = Layer(1, 2, sigmoid, sigmoid_prime)

    bias = np.array([.4, .3])
    weights = np.array([[.1], [.2]])

    Layer1.weights = weights
    Layer1.biases = bias

    Layer2 = Layer(2, 1, sigmoid, sigmoid_prime)
    bias = np.array([.7])
    weights = np.array([[.5, .6]])
    Layer2.weights = weights
    Layer2.biases = bias

    network = NeuralNetwork()
    network.add(Layer1)
    network.add(Layer2)

    x_train = [[.13], [0.42]]
    y_train = [[.90], [0.23]]

    network.fit_schocastic(x_train, y_train, 1)
    network.fit_regularization(x_train, y_train, 0)

def backpropChecker2():

    layer1 = Layer(2, 4, sigmoid, sigmoid_prime)
    layer1.weights = np.array([[.15, .4], [.1, .54], [.19, .42], [.35, .68]])
    layer1.biases = np.array([.42, .72, .01, .3])

    layer2 = Layer(4, 3, sigmoid, sigmoid_prime)
    layer2.biases = np.array([.21, .87, .03])
    layer2.weights = np.array([[.67, .14, .96, .87],[.42, .2, .32, .89],[.56, .8, .69, .09]])

    layer3 = Layer(3, 2, sigmoid, sigmoid_prime)
    layer3.biases = np.array([.04, .17])
    layer3.weights = np.array([[.87, .42, .53], [.1, .95, .69]])

    network = NeuralNetwork()
    network.add(layer1)
    network.add(layer2)
    network.add(layer3)

    x_train = [[.32, .68], [0.83, 0.02]]
    y_train = [[0.75, 0.98], [0.75, 0.28]]

    network.fit_schocastic(x_train, y_train, 1)
    network.fit_regularization(x_train, y_train, .25)


backpropChecker1()
backpropChecker2()


