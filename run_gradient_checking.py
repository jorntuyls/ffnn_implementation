from nnlayers import InputLayer, FullyConnectedLayer, LinearOutput, Activation, SoftmaxOutput
from neural_network import NeuralNetwork
import numpy as np

input_shape = (5, 10)
n_labels = 6
layers = [InputLayer(input_shape)]

layers.append(FullyConnectedLayer(
               layers[-1],
               num_units=15,
               init_stddev=0.1,
               activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
               layers[-1],
               num_units=6,
               init_stddev=0.1,
               activation_fun=Activation('sigmoid')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=n_labels,
                init_stddev=0.1,
                activation_fun=None
))
layers.append(SoftmaxOutput(layers[-1]))
nn = NeuralNetwork(layers)

# create random data
X = np.random.normal(size=input_shape)
# and random labels
Y = np.zeros((input_shape[0], n_labels))
for i in range(Y.shape[0]):
    idx = np.random.randint(n_labels)
    Y[i, idx] = 1.

# check gradients of the neural network
nn.check_gradients(X,Y)
