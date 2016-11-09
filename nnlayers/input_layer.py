# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb

from ilayer import Layer

# define a container for providing input to the network
class InputLayer(Layer):

    def __init__(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError("InputLayer requires input_shape as a tuple")
        self.input_shape = input_shape

    def output_size(self):
        return self.input_shape

    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        return output_grad
