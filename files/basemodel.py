import numpy as np
import tensorflow as tf
import pickle


###################################SQUARE_SUM###################################
def square_sum(x):
    """Computes the square root sum of the elements of a matrix/vector

    Parameters
    ----------
    x : numpy.ndarray
      Vector or Matrix

    Returns
    -------
    float
      The square roof of the elements of a matrix or vector
    """

    return tf.reduce_sum(tf.square(x))


###################################COUNT_VARS###################################
def count_vars(x):
    """Returns the number of variables in a vector/matrix

    Parameters
    ----------
    x : numpy.ndarray
      Vector or Matrix

    Returns
    -------
    int
      The number of variables in a vector/matrix
    """

    return tf.reshape(x, (-1)).shape[0]


###############################COUNT_TRAIN_VARS#################################
def count_train_vars(model):
    """ Returns the number trainable parameters of a neural network model

    Parameters
    ----------
    model : tf.Module subclass
      A neural network model

    Returns
    -------
    int
      Number of trainable parameters of a model
    """

    train_var = 0
    for var in model.trainable_variables:
        train_var += count_vars(var)
    return train_var


################################################################################
#                               BASE MODEL                                     #
################################################################################
class BaseModel(tf.Module):
    """
    A class that implements methods common for any neural network. It is designed
    to be sublcassed to create neural networks with the desired architecture.

    ...

    Attributes
    ----------
    history : dict
      Dictionary with different metrics or losses

    Methods
    -------
    predict(self, data)
      Predicts the target given an input

    loss(self, X,Y, **kwargs)
      Returns the loss that is a linear combination of the MSE and the L2
      regularization loss

    reg_loss(self)
      Returns the L2 regularization loss

    mse(self, Y_pred, Y_true)
      Returns the Mean Squarred Error (MSE)

    train(self, X, Y, **kwargs)
      Performs a minimization step of the loss by updating the trainable variables

    fit(self, X,Y, epochs=100, validation=None, print_every=1, **kwargs)
      Performs several training steps

    def save(self, filename)
      Saves the dictionary with all attributes in a file using pickle

    def load(self, filename)
      Loads the  dictionary with all attributes from a file using pickle
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

    def predict(self, data):
        """Predicts the target given an input

        Parameters
        ----------
        data : numpy.ndarray
          Matrix with inputs

        Returns
        -------
        numpy.ndarray
          Target prediction
        """

        return self(data).numpy()

    def loss(self, X, Y, **kwargs):
        """Returns the loss that is a linear combination of the MSE and the L2
        regularization loss

        Parameters
        ----------
        X : numpy.ndarray or list
          Matrix/list with inputs

        Y : numpy.ndarray or list
          Matrix/list with targets

        Returns
        -------
        float
          Loss
        """

        alpha = kwargs.get('alpha', 1)
        beta = kwargs.get('beta', 0)
        if type(X) is list:
            mse = np.sum([self.mse(self(x), y) for x, y in zip(X, Y)])
        else:
            mse = self.mse(self(X), Y)
        return alpha * mse + beta * self.reg_loss()

    def reg_loss(self):
        """Returns the L2 regularization loss

        Returns
        -------
        float
            Loss
        """
        reg_loss = 0
        for var in self.trainable_variables:
            reg_loss += square_sum(var)
        return reg_loss

    def mse(self, Y_pred, Y_true):
        """Returns the Mean Squarred Error (MSE)

        Parameters
        ----------
        Y_pred : numpy.ndarray or list
          Prediction of the target

        Y_true : numpy.ndarray or list
          Target or ground truth

        Returns
        -------
        float
          MSE Loss
        """

        return square_sum(Y_pred - Y_true) / count_vars(Y_pred)

    def train(self, X, Y, **kwargs):
        """Performs a minimization step of the loss by updating the trainable
        variables

        Parameters
        ----------
        X : numpy.ndarray or list
          Matrix/list with inputs

        Y : numpy.ndarray or list
          Matrix/list with targets
        """

        alpha = kwargs.get('alpha', 1)
        beta = kwargs.get('beta', 0)
        opt = tf.keras.optimizers.SGD(learning_rate=kwargs.get('learning_rate', 0.1))
        loss_f = lambda: self.loss(X, Y, **kwargs)
        opt.minimize(loss_f, self.vars)

    def fit(self, X, Y, epochs=100, validation=None, print_every=1, **kwargs):
        """Performs several training steps

        Parameters
        ----------
        X : numpy.ndarray or list
          Matrix/list with inputs

        Y : numpy.ndarray or list
          Matrix/list with targets

        epochs : int, optional
          Number of training steps

        validation : None or tuple, optional
         If it is a tuple it will calculate metrics on the validation data. The
         tuple must be (X_validation, Y_validation). Default value is None and
         therefore it does not perform anything

        print_every : int, optional
          It will print the metrics every print_every number of epochs
        """

        loss_val = 0
        if validation:
            X_val, Y_val = validation
        for epoch in range(1, epochs + 1):
            if type(X) is list:
                for x, y in zip(X, Y):
                    self.train(x, y, **kwargs)
            else:
                self.train(X, Y, **kwargs)
            loss = self.loss(X, Y, **kwargs).numpy()
            self.history['loss'].append(loss)
            if validation:
                loss_val = self.loss(X_val, Y_val, **kwargs).numpy()
                self.history['validation'].append(loss_val)
            if epoch % print_every == 0: print(
                f'Epoch({epoch: 4d}/{epochs: 4d}) Loss:{loss:1.4e} Validation Loss:{loss_val:1.4e}')

    def save(self, filename):
        """Saves the dictionary with all attributes in a file using pickle

        Parameters
        ----------
        filename : str
          Name of the file without the extension. The extension '.pickle' is added
        """

        with open(f'{filename}.pickle', 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load(self, filename):
        """Loads the  dictionary with all attributes  from a file using
        pickle

        Parameters
        ----------
        filename : str
          Name of the file without the extension. The extension '.pickle' is added
        """

        with open(f'{filename}.pickle', 'rb') as handle:
            self.__dict__.update(pickle.load(handle))
        return

