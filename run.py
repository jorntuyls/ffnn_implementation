from nnlayers import InputLayer, FullyConnectedLayer, LinearOutput, Activation, SoftmaxOutput
from neural_network import NeuralNetwork
import numpy as np
from mnist import mnist

# load
Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
X_val, y_val = Dval
X_test, y_test = Dtest
# Downsample training data to make it a bit faster for testing this code
n_train_samples = 50000
train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
X_train = X_train[train_idxs]
y_train = y_train[train_idxs]
print("X_val shape: {}".format(np.shape(X_val)))
print("y_val shape: {}".format(np.shape(y_val)))
print("X_test shape: {}".format(np.shape(X_test)))
print("y_test shape: {}".format(np.shape(y_test)))
print("X_train shape: {}".format(np.shape(X_train)))
print("y_train shape: {}".format(np.shape(y_train)))

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
print("Reshaped X_train size: {}".format(X_train.shape))

import time

# Setup a small MLP / Neural Network
# we can set the first shape to None here to indicate that
# we will input a variable number inputs to the network
input_shape = (None, 28*28)
layers = [InputLayer(input_shape)]
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=200,
                init_stddev=0.01,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=56,
                init_stddev=0.01,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=10,
                init_stddev=0.01,
                # last layer has no nonlinearity
                # (softmax will be applied in the output layer)
                activation_fun=None
))
layers.append(SoftmaxOutput(layers[-1]))

nn = NeuralNetwork(layers)
# Train neural network
t0 = time.time()
v = nn.train(X_train, y_train, learning_rate=np.power(10,-0.2046182), #np.power(10,-0.32434927)
         max_epochs=40, batch_size=64, descent_type="sgd", y_one_hot=True, X_val=X_val, Y_val=y_val)
t1 = time.time()
print('Duration: {:.1f}s'.format(t1-t0))
