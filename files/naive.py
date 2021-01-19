import numpy as np
import tensorflow as tf
from basemodel import *

################################################################################
#                                     NAIVE                                    #
################################################################################
class Naive(BaseModel):
    """
    A class that implements a simple perceptron (it can be made with one layer or
    multiple)

    ...

    Attributes
    ----------
    _name : str
      Name of the model

    output_var : str
      Kind of output that the model outputs. Possibilities: 'R', 'V', 'A'

    vars : dict
      Dictionary that contains the different trainable variables of the model

    n_body : int
      Number of agents/particles the last call of the model had

    layer_list : list of ints
      List with the size of the different layers of the perceptron

    n_weights : int
      Number of layers of the percetpron

    Methods
    -------
    __call__(self, data)
      Returns an updated kinematic variable of the agents/particles of the system
    """

    def __init__(self, n_body, hidden_layers=[100], output_var='V', name='Naive', load=None, **kwargs):
        """Initializes variables and weights of the model

        Parameters
        ----------
        n_body : int
          Number of objects/agents

        hidden_layers : list of ints, optional
          List with the hidden layer sizes. It can be an empty list whichs means
          there are no hidden layers

        output_var : str, optional
          Kind of output that the model outputs. Possibilities: 'R', 'V', 'A'

        name : str
          Name of the model
        """

        super().__init__(**kwargs)
        if load:
            self.load(filename=load)
        else:
            self.history = {'loss': [], 'validation': []}
            self._name = name
            self.output_var = output_var
            self.n_body = n_body
            self.layer_list = [n_body * 4] + hidden_layers + [2]
            list_in = self.layer_list[:-1]
            list_out = self.layer_list[1:]
            self.n_weights = len(list_in)
            self.vars = {}

            for i, (size_in, size_out) in enumerate(zip(list_in, list_out)):
                w_name = f'w_{i}'
                b_name = f'b_{i}'
                self.vars[w_name] = tf.Variable(tf.random.normal([size_in, size_out], stddev=0.5), name=w_name,
                                                dtype=tf.float32)
                # self.vars[w_name] =  tf.Variable( np.zeros([size_in, size_out]), name=w_name , dtype=tf.float32)
                self.vars[b_name] = tf.Variable(tf.random.normal([size_out, ], stddev=0.00), name=b_name,
                                                dtype=tf.float32)
                # self.vars[b_name] = tf.Variable( np.zeros([size_out, ]), name=b_name , dtype=tf.float32)

    def __call__(self, X):
        """ Applies and returns the marhalling function of  the Interaction Network

        Parameters
        ----------
        X : numpy.ndarray
          Feature matrix

        Returns
        -------
        numpy.ndarray
          Matrix with the targets that correspond to the kinematic
          variable output_var
        """
        X = X.astype('float32')
        t_steps, _, _ = X.shape
        X = X.reshape(t_steps, -1).repeat(self.n_body, 0).reshape(t_steps, self.n_body, self.n_body * 4)
        for i in range(self.n_body):
            x_aux = X[:, i, i * 4: (i + 1) * 4].copy()
            X[:, i, i * 4: (i + 1) * 4] = X[:, i, 0:4]
            X[:, i, 0:4] = x_aux
        for i in range(self.n_weights):
            w = self.vars[f'w_{i}']
            b = self.vars[f'b_{i}']
            if i == (self.n_weights - 1):
                X = tf.matmul(X, w) + b
            else:
                X = tf.nn.relu(tf.matmul(X, w) + b)
        return X


