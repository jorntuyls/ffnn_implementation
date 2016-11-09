# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb

# define a base class for loss outputs
# an output layer can then simply be derived
# from both Layer and Loss
class Loss(object):

    def loss(self, output, output_net):
        """ Calculate mean loss given real output and network output. """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def input_grad(self, output, output_net):
        """ Calculate input gradient real output and network output. """
        raise NotImplementedError('This is an interface class, please use a derived instance')
