# This code shows how the Bayesian hyperparameter optimizer RoBo is used
#   to optmize learning rate and number of hidden units of the feedforward
#   neural network
# !! First install RoBo
# RoBo can be found on following github page: https://github.com/automl/RoBO
import numpy as np
from robo.fmin import fmin
from ffnn_implementation.nnlayers import InputLayer, FullyConnectedLayer, LinearOutput, Activation, SoftmaxOutput
from ffnn_implementation.neural_network import NeuralNetwork
import numpy as np
from ffnn_implementation.mnist import mnist

# load mnist data
Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
X_val, y_val = Dval
# Downsample training data to make it a bit faster for testing this code
n_train_samples = 10000
train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
X_train = X_train[train_idxs]
y_train = y_train[train_idxs]

# Reshape X_train and X_val
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# Define objective function
def objective_function(x):
    print("Create neural network")
    print(x)
    input_shape = (None, 28*28)
    layers = [InputLayer(input_shape)]
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=int(x[0][1]),
            init_stddev=0.01,
            activation_fun=Activation('relu')
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=int(x[0][2]),
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

    print("Evaluation of objective function")
    v = nn.train(X_train, y_train, learning_rate=np.power(10,float(x[0][0])),
         max_epochs=20, batch_size=64, descent_type="sgd", y_one_hot=True, X_val=X_val, Y_val=y_val)
    return np.array([[v]])

# define upper and lower boundaries
X_lower = np.array([-6,0,0])
X_upper = np.array([0,1001,1001])

# run optimizer
import time
t0 = time.time()
x_best, fval = fmin(objective_function, X_lower, X_upper)
t1 = time.time()
print("Calculation time: {}".format(t1-t0))
print(x_best, fval)
