import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import DataPrep as dp
import pandas as pd
from sklearn.neural_network import MLPClassifier


data = pd.read_excel('output.xlsx')
d, x_1, x_2, data_y = dp.prepare_data_feature_selection(data, 13750, 86190, 1, 1)


X = np.array(data_y)[0:10,1:3]
y = np.array(x_1)[0:10]

X = X.astype('float64')
y = y.astype('float64')


X_new = []
tmp = []


for j in range(X.shape[0]):
    for i in range(X.shape[1]):
        tmp.append(X[j][i])

    X_new.append(tmp)
    tmp = []

y_new = []

for i in range(y.shape[0]):
    y_new.append(float(y[i]))




clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X_new, y_new)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')


print(clf.predict(np.array(data_y)[501:502,1:]))


"""
# input data
data = pd.read_excel('output.xlsx')
d, x_1, x_2, data_y = dp.prepare_data_feature_selection(data, 13750, 86190, 1, 1)

print(data_y.shape)
print(x_1.shape)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

X = data_y[1500:2000]
y = x_1[1500:2000]

y_new = []


for i in range(len(y)):
    y_new.append(y.to_numpy()[i])

clf.fit(X, y_new)

clf.predict(np.array(y[100:200]))







inputs = np.array(y[:50000])
# output data
outputs = (np.array(x_1[:50000])).reshape(-1,1)
# print(inputs.shape)


# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity

        arr = []
        for i in range(inputs.shape[1]):
            arr.append([0.5])

        arr = np.array(arr)
        # arr.transpose(axis=1)
        self.weights = arr
        # print('test2', self.weights)
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=5000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

            if epoch % 500 == 0:
                print(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


# create neural network
print(inputs)
print('-----------------------------------------------------------')
print(outputs)

NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

# create two new examples to predict
example = (np.array(y[0:1200]))
example_2 = (np.array(y[0:1200]))

# print the predictions for both examples
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
"""
