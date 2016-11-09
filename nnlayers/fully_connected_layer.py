# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb

from ilayer import Layer
from iparameterized import Parameterized
from activation import Activation

import numpy as np

class FullyConnectedLayer(Layer, Parameterized):
    """ A standard fully connected hidden layer, as discussed in the lecture.
    """

    def __init__(self, input_layer, num_units,
                 init_stddev, activation_fun=Activation('relu')):
        self.num_units = num_units
        self.activation_fun = activation_fun
        # the input shape will be of size (batch_size, num_units_prev)
        # where num_units_prev is the number of units in the input
        # (previous) layer
        self.input_shape = input_layer.output_size()
        # TODO ################################
        # TODO: implement weight initialization
        # TODO ################################
        # this is the weight matrix it should have shape: (num_units_prev, num_units)
        # use normal distrbution with mean 0 and stdv = init_stddev
        self.W = np.random.normal(0, init_stddev, (self.input_shape[1], num_units)) #FIXME
        # and this is the bias vector of shape: (num_units)
        self.b = np.random.normal(0, init_stddev, (num_units,))  #FIXME
        # create dummy variables for parameter gradients
        # no need to change these here!
        self.dW = None
        self.db = None

    def output_size(self):
        return (self.input_shape[0], self.num_units)

    def fprop(self, input):
        # TODO ################################################
        # TODO: implement forward propagation
        # NOTE: you should also handle the case were
        #       activation_fun is None (meaning no activation)
        #       then this is simply a linear layer
        # TODO ################################################
        # you again want to cache the last_input for the bprop
        # implementation below!
        self.last_input = input
        # FIXME
        # print("begin fprop")
        #print(input)
        #print(self.W)
        #print(self.b)
        # print(input.dot(self.W))
        z = np.dot(input,self.W) + self.b
        # print(z)
        # print("end fprop")
        if self.activation_fun:
            return self.activation_fun.fprop(z)
        return z

    def bprop(self, output_grad):
        """ Calculate input gradient (backpropagation). """
        # TODO ################################
        # TODO: implement backward propagation
        # TODO ###############################

        # HINT: you may have to divide the weights by n
        #       to make gradient checking work
        #       (since you want to divide the loss by number of inputs)
        #print(output_grad.shape)
        n = output_grad.shape[0]
        # accumulate gradient wrt. the parameters first
        # we will need to store these to later update
        # the network after a few forward backward passes
        # the gradient wrt. W should be stored as self.dW
        # the gradient wrt. b should be stored as self.db

        # convert the gradient on the layer's output into a gradient in the
        #       prenonlinearity activation
        #print("begin bprop")
        #print(self.W)
        #print(self.b)

        if self.activation_fun:
            output_grad = self.activation_fun.bprop(output_grad)
        # TODO For now there is no regularization term used
        self.dW = self.last_input.transpose().dot(output_grad) / n #FIXME
        self.db = np.sum(output_grad, axis=0) / n #FIXME
        #print(self.dW.shape)
        #print(self.db.shape)

        # the gradient wrt. the input should be calculated here
        grad_input = output_grad.dot(self.W.transpose())
        #print(grad_input.shape)
        #print(self.last_input.shape)
        #print("end bprop")
        return grad_input

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db

    def update_params(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
        #print("WWWWWWWWWWW")
        #print(self.W)
        #print("bbbbbbbbbbb")
        #print(self.b)
