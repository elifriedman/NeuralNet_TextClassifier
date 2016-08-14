"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T
from tools.gradupdates import *


# start-snippet-1
class Layer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.rng = rng
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.n_in = n_in
        self.n_out = n_out
        
        # parameters of the model
        self.params = [self.W, self.b]



    def train_function(self,cost,params=None,inputs = [],outputs = [], givens = [], update_type="sgd", paramlist=None):
        """
        Create the training function for this network. You can use it by calling RNN.train()

        :type cost: theano.tensor.TensorVariable
        :param cost: a symbolic tensor variable representing the cost function to minimize

        :type learning_rate: float
        :param learning_rate: scaling factor for gradient used in gradient descent (We only support fixed learning_rates right now)

        :type params: list of theano.tensor.TensorVariables
        :param params: parameters to be updated by gradient descent, defaults to self.params

        :type inputs: list of theano.tensor.TensorVariables
        :param inputs: inputs to be input into the function

        :type outputs: list of theano.tensor.TensorVariables
        :param outputs: outputs to be output from the function

        :type givens: list of theano.tensor.TensorVariables
        :param givens: givens to be given to the function
        """
        if params == None:
            params = self.params
        grad = T.grad(cost,params)

        if update_type == "sgd" or update_type=="" or update_type==None:
          self.train = SGD(params, grad, paramlist[0],inputs=inputs,outputs=outputs,givens=givens)
        elif update_type == "momentum":
          self.train = SGD_momentum(params,grad,paramlist[0],paramlist[1],
                                      inputs=inputs,outputs=outputs,givens=givens)
        elif update_type == "ada":
          self.train = ada_grad(params,grad,paramlist[0],inputs=inputs,outputs=outputs,givens=givens)
        elif update_type == 'adam':
          lr = 0.0002 if len(paramlist) < 1 else paramlist[0]
          b1 = 0.1 if len(paramlist) < 2 else paramlist[1]
          b2 = 0.001 if len(paramlist) < 3 else paramlist[2]
          e = 1e-8 if len(paramlist) < 4 else paramlist[3]
          self.train = adam(params,grad, lr=lr, b1=b1, b2=b2, e=e,inputs=inputs,outputs=outputs,givens=givens)
        else:
          raise ValueError('Not a valid update_type. Valid update_types are' +
                            '[sgd,momentum,ada] for sgd, sgd+momentum, and adagrad')

    def test_function(self, inputs = [],outputs = [], givens = []):
        """
        Create the testing function for this network. You can use it by calling RNN.test()

        :type inputs: list of theano.tensor.TensorVariables
        :param inputs: inputs to be input into the function

        :type outputs: list of theano.tensor.TensorVariables
        :param outputs: outputs to be output from the function

        :type givens: list of theano.tensor.TensorVariables
        :param givens: givens to be given to the function
        """
        self.test = theano.function(
            inputs = inputs,
            outputs = outputs,
            givens = givens
        )


