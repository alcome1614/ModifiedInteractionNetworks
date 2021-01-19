import numpy as np
import tensorflow as tf
from basemodel import *


##########################CREATE_R_MTX##########################################
def create_R_mtx(n_body, self_edges=True):
    """Returns the R_r, R_s matrices of the Interaction network which index the
    receiver and sender objects

    Parameters
    ----------
    n_body : int
      Number of nodes

    self_edges : bool, optional
      If True it only creates edges between different nodes. If False it will
      include edges that go from one node to itself

    Returns
    -------
    numpy.ndarray
      The matrix R_r which indexes the receiver objects

    numpy.ndarray
      The matrix R_s which indexes the sender objects
    """

    n_edges = n_body * n_body
    if not self_edges:
        n_edges = n_edges - n_body
    R_s = np.zeros((n_body, n_edges), dtype=np.int32)
    R_r = np.zeros((n_body, n_edges), dtype=np.int32)
    edge_index = 0
    for i in range(n_body):
        for j in range(n_body):
            if i < j or i > j:
                R_s[i, edge_index] = 1
                R_r[j, edge_index] = 1
                edge_index += 1
            elif i == j and self_edges:
                R_s[i, edge_index] = 1
                R_r[j, edge_index] = 1
                edge_index += 1

    return tf.convert_to_tensor(R_s, dtype=tf.float32), tf.convert_to_tensor(R_r, dtype=tf.float32)


#####################################M##########################################
def m(O, R_s, R_r):
    """ Applies and returns the marhalling function of  the Interaction Network

    Parameters
    ----------
    O : numpy.ndarray
      Object matrix

    R_s : numpy.ndarray
      Binary matrix that indexes the sender objects

    R_r : numpy.ndarray
      Binary matrix that indexes the receiver objects

    Returns
    -------
    numpy.ndarray
      Matrix B of features of the relations
    """

    B = tf.matmul(O, R_s - R_r)
    return B


#####################################A##########################################
def a(O, R_r, E, aggregator='sum', include_v=True):
    """ Applies and returns the aggregating function of the Interaction Network

    Parameters
    ----------
    O : numpy.ndarray
      Object matrix

    R_r : numpy.ndarray
      Binary matrix that indexes the receiver objects

    E : numpy.ndarray
      Predicted effects matrix

    aggregator : str, optinal
      Kind of aggregation to apply. By default it is 'sum' of edges effects.
      Another option is the 'mean'

    inlcude_v : bool, optional
      If to include velocity in the concatenation or not. Default value is True.

    Returns
    -------
    numpy.ndarray
      Matrix B of features of the relations
    """

    E_bar = tf.matmul(E, tf.transpose(R_r, [1, 0]))
    if aggregator == 'mean':
        E_bar = E_bar / tf.reduce_sum(R_r, axis=1)
    if include_v == False:
        C = E_bar
        return C
    O_2 = O[:, 2:4, :]
    C = tf.concat([O_2, E_bar], axis=1)
    return C


################################################################################
#                          INTERACTION NETWORK                                 #
################################################################################
class InteractionNetwork(BaseModel):
    """
    A class that implements an Interaction Network with some modifications that
    can be set

    ...

    Attributes
    ----------
    _name : str
      Name of the model

    include_v : bool
      If to include velocity in the aggregation function

    aggregator : str
      Kind of aggregation to apply. By default it is 'sum' of edges effects.
      Another option is the 'mean'

    self_edges : bool
      If True it only creates edges between different nodes. If False it will
      include edges that go from one node to itself. Used in the create_R_mtx
      function

    output_var : str
      Kind of output that the model outputs. Possibilities: 'R', 'V', 'A'

    vars : dict
      Dictionary that contains the different trainable variables of the model

    n_body : int
      Number of agents/particles the last call of the model had

    n_edges : int
      Number of edges or links the last of the model had

    R_r : numpy.ndarray
      The matrix which indexes the receiver objects

    R_s : numpy.ndarray
      The matrix which indexes the sender objects

    D_E : int
      Dimension of the effects inferred

    R_layer_list : list of ints
      List with the size of the different layers of the relational perceptron

    O_layer_list : list of ints
      List with the size of the different layers of the object perceptron

    R_n_weights : int
      Number of layers of the relational percetpron

    O_n_weights : int
      Number of layers of the object percetpron

    Methods
    -------
    phi_R(self, B)
      Relational perceptron that outputs effects of interaction

    phi_O(self, C)
      Object perceptron that outputs kinematic variables of the objects

    __call__(self, data)
      Returns an updated kinematic variable of the agents/particles of the system
    """

    def __init__(self, D_E=100, R_hidden_layers=[100], O_hidden_layers=[100], name='InteractionNetwork', output_var='V',
                 include_v=True, aggregator='sum', self_edges=True, load=None, **kwargs):
        """
        Parameters
        ----------
        D_E : int, optional
          Dimension of the effects inferred

        R_hidden_layers : list of ints, optional
          List with the size of the different hidden layers of the relational
          perceptron. It can be an empty list meaning it has no hidden layers and it
          becomes a single-layer perceptron

        O_hidden_layers : list of ints, optional
          List with the size of the different hidden layers of the object
          perceptron. It can be an empty list meaning it has no hidden layers and it
          becomes a single-layer perceptron

        name : str, optional
          Name of the model

        include_v: bool, optional
          If to include velocity in the aggregation function. Default value True

        aggregator : str, optional
          Kind of aggregation to apply. By default it is 'sum' of edges effects.
          Another option is the 'mean'

        self_edges : bool, optional
          If True it only creates edges between different nodes. If False it will
          include edges that go from one node to itself. Used in the create_R_mtx
          function

        load : str or None, optional
          Name of the file from which to load the attributes. If None then there
          will be no loading from file
        """

        super().__init__(**kwargs)
        if load:
            self.load(load)
        else:
            self.history = {'loss': [], 'validation': []}
            self._name = name
            self.include_v = include_v
            self.aggregator = aggregator
            self.self_edges = self_edges
            self.output_var = output_var
            self.vars = {}
            self.n_body = None
            self.n_edges = None
            self.R_r = None
            self.R_s = None
            self.D_E = D_E
            self.R_layer_list = [4] + R_hidden_layers + [D_E]
            # self.R_layer_list = [2] + R_hidden_layers + [D_E]
            if include_v:
                self.O_layer_list = [2 + D_E] + O_hidden_layers + [2]
            else:
                self.O_layer_list = [D_E] + O_hidden_layers + [2]
            self.R_n_weights = len(self.R_layer_list) - 1
            self.O_n_weights = len(self.O_layer_list) - 1

            for i, (size_in, size_out) in enumerate(zip(self.R_layer_list[:-1], self.R_layer_list[1:])):
                self.vars[f'R_w_{i}'] = tf.Variable(tf.random.normal([size_in, size_out], stddev=0.1), name=f'R_w_{i}')
                # self.vars[f'R_w_{i}'] = tf.Variable( np.array([[0,0.],[0,0],[1.,0],[0,1.]]), name=f'O_w_{i}' , dtype=tf.float32)
                # self.vars[f'R_b_{i}'] = tf.Variable( tf.random.normal([size_out, ], stddev=0.1), name=f'R_b_{i}' )
                # self.vars[f'R_w_{i}'] = tf.Variable( np.zeros([size_in, size_out]), name=f'R_w_{i}' , dtype=tf.float32)
                self.vars[f'R_b_{i}'] = tf.Variable(np.zeros([size_out, ]), name=f'R_b_{i}', dtype=tf.float32)
                # self.vars[f'R_b_{i}'] = tf.Variable( np.array([1e-3, 0]), name=f'R_b_{i}' , dtype=tf.float32)

            for i, (size_in, size_out) in enumerate(zip(self.O_layer_list[:-1], self.O_layer_list[1:])):
                self.vars[f'O_w_{i}'] = tf.Variable(tf.random.normal([size_in, size_out], stddev=0.1), name=f'O_w_{i}')
                # self.vars[f'O_b_{i}'] = tf.Variable( tf.random.normal([size_out, ], stddev=0.1), name=f'O_b_{i}' )
                # self.vars[f'O_w_{i}'] = tf.Variable( np.zeros([size_in, size_out]), name=f'O_w_{i}' , dtype=tf.float32)
                # self.vars[f'O_w_{i}'] = tf.Variable( np.array([[1.1,0.1],[0.1,1.1],[0,0],[0,0]]), name=f'O_w_{i}' , dtype=tf.float32)
                self.vars[f'O_b_{i}'] = tf.Variable(np.zeros([size_out, ]), name=f'O_b_{i}', dtype=tf.float32)

    def phi_R(self, B):
        """Relational perceptron that outputs effects of interaction

        Parameters
        ----------
        B : numpy.ndarray
          Matrix of relation features

        Returns
        -------
        numpy.ndarray
          Matrix E of effects inferred from the relations
        """

        n_batches, _, n_edges = B.shape
        result = tf.transpose(B, [0, 2, 1])
        for i in range(self.R_n_weights - 1):
            result = tf.nn.relu(tf.matmul(result, self.vars[f'R_w_{i}']) + self.vars[f'R_b_{i}'])
        weights = self.R_n_weights
        result = tf.matmul(result, self.vars[f'R_w_{weights - 1}']) + self.vars[f'R_b_{weights - 1}']
        E = tf.transpose(result, [0, 2, 1])
        return E

    def phi_O(self, C):
        """Object perceptron that outputs kinematic variables of the objects

        Parameters
        ----------
        C : numpy.ndarray
          Matrix of object features because of it and because of its environment

        Returns
        -------
        numpy.ndarray
          Matrix with the kinematic variables of the agents/particles
      """

        n_batches, _, n_body = C.shape
        result = tf.transpose(C, [0, 2, 1])
        for i in range(self.O_n_weights - 1):
            result = tf.nn.relu(tf.matmul(result, self.vars[f'O_w_{i}']) + self.vars[f'O_b_{i}'])
        weights = self.O_n_weights
        result = tf.matmul(result, self.vars[f'O_w_{weights - 1}']) + self.vars[f'O_b_{weights - 1}']
        P = result
        P = tf.reshape(result, (n_batches, n_body, 2))
        return P

    def __call__(self, data):
        """Returns an updated kinematic variable of the agents/particles of the
        system

        Parameters
        ----------
        data : numpy.ndarray
          Matrix input of the model with information about the agents/particles

        Returns
        -------
        numpy.ndarray
          Matrix with the kinematic variables of the agents/particles
        """

        n_batches, n_body, _ = data.shape
        O = tf.transpose(tf.convert_to_tensor(data, dtype=tf.float32), [0, 2, 1])
        # O = tf.transpose(tf.convert_to_tensor( data[:,:,0:2], dtype=tf.float32), [0,2,1])
        if n_body != self.n_body:
            self.n_body = n_body
            self.R_s, self.R_r = create_R_mtx(n_body, self_edges=self.self_edges)
            self.n_edges = self.R_s.shape[1]
        B = m(O, self.R_s, self.R_r)
        E = self.phi_R(B)
        # self.E = E
        C = a(O, self.R_r, E, aggregator=self.aggregator, include_v=self.include_v)
        P = self.phi_O(C)
        return P

