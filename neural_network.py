# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb

import numpy as np
import copy
from nnlayers.utility_functions import one_hot, unhot
from nnlayers.iparameterized import Parameterized

class NeuralNetwork:
    """ Our Neural Network container class.
    """
    def __init__(self, layers):
        self.layers = layers

    def _loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        # TODO ##########################################
        # TODO: implement forward pass through all layers
        # TODO ##########################################
        Y_pred = X
        for layer in self.layers:
            Y_pred = layer.fprop(Y_pred)
        return Y_pred

    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through
            the complete network up to layer 'upto'
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        # TODO ##########################################
        # TODO: implement backward pass through all layers
        # TODO ##########################################
        # -2 because -1 is the output_layer and is already done
        for i in range(len(self.layers)-2, upto-1, -1):
            next_grad = self.layers[i].bprop(next_grad)
        return next_grad

    def classification_error(self, X, Y):
        """ Calculate error on the given data
            assuming they are classes that should be predicted.
        """
        Y_pred = unhot(self.predict(X))
        error = Y_pred != Y
        return np.mean(error)

    def sgd_epoch(self, X, Y, learning_rate, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        for b in range(n_batches):
            # TODO #####################################
            # Implement stochastic gradient descent here
            # TODO #####################################
            # start by extracting a batch from X and Y
            # (you can assume the inputs are already shuffled)
            X_sample = X[b*batch_size:(b+1)*batch_size,:]
            Y_sample = Y[b*batch_size:(b+1)*batch_size,:]
            # TODO: then forward and backward propagation + updates
            # HINT: layer.params() returns parameters *by reference*
            #       so you can easily update in-place
            # first predict Y
            Y_pred = self.predict(X_sample)
            # then backpropagate
            self.backpropagate(Y_sample,Y_pred)
            # update parameters
            for i in range(1,len(self.layers)-1):
                self.layers[i].update_params(learning_rate)


    def gd_epoch(self, X, Y, learning_rate): # TODO: there should be a learning rate here
        # TODO ##################################################
        # Implement batch gradient descent here
        # A few hints:
        #   There are two strategies you can follow:
        #   Either shove the whole dataset throught the network
        #   at once (which can be problematic for large datasets)
        #   or run through it batch wise as in the sgd approach
        #   and accumulate the gradients for all parameters as
        #   you go through the data. Either way you should then
        #   do one gradient step after you went through the
        #   complete dataset!
        # TODO ##################################################
        # first predict Y
        Y_pred = self.predict(X)
        # then backpropagate
        self.backpropagate(Y,Y_pred)
        # update parameters
        for i in range(1,len(self.layers)-1):
            self.layers[i].update_params(learning_rate)


    def train(self, X, Y, learning_rate=0.1, max_epochs=100,
              batch_size=64, descent_type="sgd", y_one_hot=True, X_val=None, Y_val=None):
        """ Train network on the given data. """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        validation_error = 1.0
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y
        print("... starting training")
        for e in range(max_epochs+1):
            if descent_type == "sgd":
                # Comments: implementation of variable learning rate
                # For formula see http://www.deeplearningbook.org/contents/optimization.html
                alpha = e/(max_epochs * 1.5)
                var_learning_rate = (1-alpha)*learning_rate + alpha * (1 / (max_epochs * 1.5))*learning_rate
                self.sgd_epoch(X, Y_train, var_learning_rate, batch_size)
            elif descent_type == "gd":
                self.gd_epoch(X, Y_train, learning_rate)
            else:
                raise NotImplementedError("Unknown gradient descent type {}".format(descent_type))

            # Output error on the training data
            train_loss = self._loss(X, Y_train)
            train_error = self.classification_error(X, unhot(Y_train)) #TODO Was wrong
            #print('epoch {:.4f}, loss {:.4f}, train error {:.4f}'.format(e, train_loss, train_error))
            # TODO ##################################################
            # compute error on validation data:
            # simply make the function take validation data as input
            # and then compute errors here and print them
            # TODO ##################################################
            if type(X_val) is np.ndarray and type(Y_val) is np.ndarray:
                validation_loss = self._loss(X_val, one_hot(Y_val))
                validation_error = self.classification_error(X_val, Y_val)
                print('Validation: epoch {:.4f}, loss {:.4f}, validation error {:.4f}'.format(e, validation_loss, validation_error))
        print("Validation error: {}".format(validation_error))
        return validation_error

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        for l, layer in enumerate(self.layers):
            if isinstance(layer, Parameterized):
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                    # we iterate through all parameters
                    param_shape = param.shape
                    # define functions for conveniently swapping
                    # out parameters of this specific layer and
                    # computing loss and gradient with these
                    # changed parametrs
                    def output_given_params(param_new):
                        """ A function that will compute the output
                            of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # return computed loss
                        return self._loss(X, Y)

                    def grad_given_params(param_new):
                        """A function that will compute the gradient
                           of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation through the net
                        Y_pred = self.predict(X)
                        # Backpropagation of partial derivatives
                        self.backpropagate(Y, Y_pred, upto=l)
                        # return the computed gradient
                        return np.ravel(self.layers[l].grad_params()[p])

                    # let the initial parameters be the ones that
                    # are currently placed in the network and flatten them
                    # to a vector for convenient comparisons, printing etc.
                    param_init = np.ravel(np.copy(param))

                    # TODO ####################################
                    # TODO compute the gradient with respect to
                    #      the initial parameters in two ways:
                    #      1) with grad_given_params()
                    #      2) with finite differences
                    #         using output_given_params()
                    #         (as discussed in the lecture)
                    #      if your implementation is correct
                    #      both results should be epsilon close
                    #      to each other!
                    # TODO ####################################
                    epsilon = 1e-4
                    # making sure your gradient checking routine itself
                    # has no errors can be a bit tricky. To debug it
                    # you can "cheat" by using scipy which implements
                    # gradient checking exactly the way you should!
                    # To do that simply run the following here:
                    #import scipy.optimize
                    #err = scipy.optimize.check_grad(output_given_params,
                    #                               grad_given_params, param_init)
                    #loss_base = output_given_params(param_init)
                    # TODO this should hold the gradient as calculated through bprop
                    gparam_bprop = grad_given_params(param_init)
                    #print("gparam_bprop: {}".format(gparam_bprop))
                    # TODO this should hold the gradient calculated through
                    #      finite differences
                    gparam_fd = []
                    for i in range(0,len(param_init)):
                        param_init_eps_plus = copy.copy(param_init)
                        param_init_eps_min = copy.copy(param_init)
                        param_init_eps_plus[i] += epsilon
                        param_init_eps_min[i] -= epsilon
                        loss_plus_eps = output_given_params(param_init_eps_plus)
                        loss_min_eps = output_given_params(param_init_eps_min)
                        gparam_fd.append((loss_plus_eps - loss_min_eps)/(2*epsilon))
                    #print("gparam_fd: {}".format(gparam_fd))
                    # calculate difference between them
                    err = np.mean(np.abs(gparam_bprop - gparam_fd))
                    print('diff {:.2e}'.format(err))
                    assert(err < epsilon)

                    # reset the parameters to their initial values
                    param[:] = np.reshape(param_init, param_shape)
