import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import tensorflow as tf
import imageio

css_colors = mcolors.CSS4_COLORS
import pickle
import imageio


###################################FIG_TO_IMG###################################
def fig_to_img(fig):
    """Converts a figure object into an image

    Parameters
    ----------
    fig : matplotlib.figure.Figure
      Figure object

    Returns
    -------
    float or int
      The cosine of the vectors
    """

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


###################################GIF_FROM_LIST################################
def gif_from_list(img_list, filename, fps=1):
    """Creates a gif file from a list of images

    Parameters
    ----------
    img_list: list
      The list of images to make into a gif file

    filename : str
      The name of the file of the gif that will be created without extension

    fps : int, optional
      The number of frames per second of the gif( or images per second) displayed
    """

    # kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave(filename + '.gif', img_list, fps=fps)


###################################COSINE#######################################
def cosine(v1, v2):
    """Computes the cosine between two vectors.

    Parameters
    ----------
    v1 : numpy.ndarray
      vector 1, 1-dimensional

    v2 : numpy.ndarray
      vector 2, 1-dimensional

    Returns
    -------
    float or int
        the cosine of the vectors
    """

    v1_m = np.linalg.norm(v1)
    v2_m = np.linalg.norm(v2)
    if v1_m < 1e-7 or v2_m < 1e-7:
        return 1
    else:
        return v1.dot(v2) / (v1_m * v2_m)


################################VICSEK##########################################
def vicsek(R, V, **kwargs):
    """Computes the acceleration associated to a vicsek model time-step given a
    configuration of agents

    Parameters
    ----------
    R : numpy.ndarray
      (n_body, 2) matrix of positions of the agents, with n_body being the number
      of agents

    V : numpy.ndarray
      (n_body, 2) matrix of velocities of the agents, with n_body being the number
       of agents

    Returns
    -------
    numpy.ndarray
      (n_body, 2) matrix of accelerations of the agents, with n_body being the
      number of agents
    """

    dt = kwargs.get('dt', 1.)
    V_new = np.zeros(V.shape)
    cosine_vision = np.cos(kwargs.get('angle_vision', np.pi * 2) / 2)
    for i in range(len(R)):
        Ri = R[i]
        Vi = V[i]
        n_neighbours = 0
        for j in range(len(R)):
            Rj = R[j]
            Vj = V[j]
            if np.linalg.norm(Ri - Rj) <= kwargs.get('radius', 0.2) and cosine(Vi, Rj - Ri) >= cosine_vision:
                n_neighbours += 1
                V_new[i] += Vj
        V_new[i] = V_new[i] / np.float(n_neighbours)
    A = (V_new - V) / dt
    return A


##################################COULOMB#######################################
def coulomb(r1, r2, **kwargs):
    """Computes the central force that a particle exerts on another

    Parameters
    ----------
    r1 : numpy.ndarray
      Vector of position of the particle that receives the force

    r2 : numpy.ndarray
      Vector of position of the particlce that exerts the force

    Returns
    -------
    numpy.ndarray
        Vector of the force
    """

    alpha = kwargs.get('alpha', 1.)
    r = r1 - r2
    return alpha * r / np.linalg.norm(r) ** (kwargs.get('beta', 2) + 1)


############################CENTRAL_FORCE#######################################
def central_force(R, V, **kwargs):
    """Computes the accelerations that 1-unit of mass particles experience because
    of a certain configuration

    Parameters
    ----------
    R : numpy.ndarray
      (n_body, 2) matrix of positions of the agents, with n_body being the number
      of particles

    V : numpy.ndarray
      (n_body, 2) matrix of velocities of the agents, with n_body being the number
      of particles

    Returns
    -------
    numpy.ndarray
        (n_body, 2) matrix of accelerations of the agents, with n_body being the
        number of particles
    """
    A = np.zeros(R.shape)
    a_0 = kwargs.get('a_0', np.array([0, 0]))
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            Ai = coulomb(R[i], R[j], **kwargs)
            A[i] += Ai + a_0
            A[j] += -Ai + a_0
    return A


###########################ZERO_INTERACTION#####################################
def zero_interaction(R, V, **kwargs):
    """Computes the accelerations that 1-unit-of-mass particles experience which
    is null because there is no interaction

    Parameters
    ----------
    R : numpy.ndarray
      (n_body, 2) matrix of positions of the agents, with n_body being the number
      of particles/agents

    V : numpy.ndarray
      (n_body, 2) matrix of velocities of the agents, with n_body being the number
      of particles/agents

    Returns
    -------
    numpy.ndarray
        (n_body, 2) matrix of accelerations of the particle/agents
    """

    A = np.zeros(V.shape)
    return A


##########################POLYGON_INIT##########################################
def polygon_init(n, max_norm):
    """Computes a random initial configuration of n  2D-vectors such that they
    represent the vertices in a n edges regular polygon

    Parameters
    ----------
    n : int
      Number of edges and vertices of the polygon and number of vectors

    max_norm : float or int
      The distance from the center of the polygon to the vertices

    Returns
    -------
    numpy.ndarray
        (n, 2) matrix of vectors
    """

    R = np.zeros((n, 2))
    angle = np.pi * 2 / n
    mtx = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    angle_0 = np.random.rand() * 2 * np.pi
    R[0] = np.array([np.cos(angle_0), np.sin(angle_0)]) * max_norm
    for i in range(1, n):
        R[i] = np.matmul(mtx, R[i - 1])
    return R


##################################RANDOM_INIT###################################
def random_init(n, max_norm):
    """Computes a random initial configuration of n  2D-vectors such that they all
    are inside of a circle of radius max_norm

    Parameters
    ----------
    n : int
      Number of vectors

    max_norm : float or int
      Radius of the circle or maximum possible distance from the origin of
      coordinates that the vectors can have.

    Returns
    -------
    numpy.ndarray
        (n, 2) matrix of vectors
    """

    X = np.zeros((n, 2))
    angles = np.random.rand(n) * 2 * np.pi
    norms = np.random.rand(n) * max_norm
    for i, angle, norm in zip(range(n), angles, norms):
        X[i] = np.array([np.cos(angle), np.sin(angle)]) * norm
    return X


####################################MSE#########################################
def mse(a, b):
    """Computes the Mean Squared Error (MSE) of two vectors or matrices

    Parameters
    ----------
    a : numpy.ndarray
      First vector or array

    b : numpy.ndarray
      Second vector or array

    Returns
    -------
    float
      MSE of the inputs
    """
    n = np.sum(np.zeros(a.shape) + 1)
    return np.sum((a - b) ** 2) / n


################################################################################
#                                   SCENE                                      #
################################################################################
class Scene():
    """
    A class that stores information on a configuration of particles/agents and is
    capable of simulating its behaviour using numerical solutions or using machine
    learning techniques

    ...

    Attributes
    ----------
    n_body : int
        Number of agents/particles of the Scene

    t_steps: int
      Number timesteps or frames of the Scene

    dt: int
      Time increment for the simulations

    RVA : dict
      Contains different entries each of them is a matrix with the kinematic
      variables (position, velocity and acceleration) of the particles/agents at
      different timesteps. The 'truth' entry corresponds to the results produced
      using numerical integration of the equations of motion. The other entries
      have the name of the model used to predict the next timesteps.

    MSE : dict
      Contains different entries each of them contains time lists of value of the
      Mean Squared Error (MSE)  of the position (R) of theparticles/agents at
      every instant of time with respect to the 'truth' entry which corresponds to
      the ground truth. T. The entries have the name of the model used to predict
      the next timesteps.

    r : float
      Maximum distance from the center of the particles generated

    v : float
      Maximum norm of the velocities generated

    Methods
    -------
    update_value(self, who, var, new_value, when)
      Updates the value of some elements of RVA

    get_value(self, who, var, when=None)
      Returns values of some elements of RVA

    get_target(self, var)
      Returns the target matrix from the ground truth

    get_features(self)
      Returns the feature matrix the from ground truth

    simulate(self, interaction, t_steps, **kwargs)
      Computes the configurations of the particles/agents according to the
      interaction selected using a numerical method at different timesteps by
      integrating the equations of motion

    predict_instant(self, model, who, t)
      Computes a new configuration of the particles/agents using a machine
      learning model for a timestep

    compute_mse(self, who1, who2, t)
      Computes the Mean Squared Error (MSE) of the position between to instant
      frames of two sources

    predict_trajectory(self, model)
       Computes a new configuration of the particles/agents using a machine
       learning model (or multiple) for several timesteps and the MSE with respect
       to the ground truth

    plot_instant(self, t, who, ax, **kwargs)
      Plots the configuration of the system at a selected timestep

    plot_step_comparison(self, model, ax, t=0, **kwargs)
      Plots a comparison between the calculations done using a numerical method
      and the results obtained by inference using a machine learning model at one
      timestep

    plot_trajectory(self, who, ax, **kwargs)
      Plots the configuration of the system at the diferent times to show the
      trajectories of the agents/particles

    plot_trajectory_comparison
      Plots a side-by-side comparison between the calculations done using a
      numerical method and the results obtained by inference using a machine
      learning model for a trajectory

    trajectory_to_gif(self, who, filename, fps=1, **kwargs)
      Creates a gif file of the different configurations of the system at
      different timesteps
    """

    def __init__(self, n_body, R_init, V_init, r=0.25, v=0.1, dt=1.):
        """
        Parameters
        ----------
        n_body : int
            Number of agents/particles of the Scene

        t_steps: int
          Number timesteps or frames of the Scene

        dt: int
          Time increment for the simulations

        RVA : dict
          Contains different entries each of them is a matrix with the kinematic
          variables (position, velocity and acceleration) of the particles/agents at
          different timesteps. The 'truth' entry corresponds to the results produced
          using numerical integration of the equations of motion. The other entries
          have the name of the model used to predict the next timesteps.

        r : float, optional
          Maximum distance from the center of the particles generated

        v : float, optional
          Maximum norm of the velocities generated

        dt : float, optional
          Increment of time
        """

        self.n_body = n_body
        self.t_steps = 0
        self.dt = dt
        self.RVA = {}
        self.RVA['truth'] = np.zeros((1, n_body, 6))
        R = R_init(self.n_body, max_norm=r)
        V = V_init(self.n_body, max_norm=v)
        self.update_value(who='truth', var='R', new_value=R, when=0)
        self.update_value(who='truth', var='V', new_value=V, when=0)
        self.MSE = {}
        self.r = r
        self.v = v

    def update_value(self, who, var, new_value, when):
        """Updates the value of some elements of RVA

        Parameters
        ----------
        who : str
          The element of the dictionary to update. If 'Truth' it refers to the
          matrix of values generated intregraing the equations of motion and it is
          the ground truth. Otherwise it must have the name of the machine lerarning
          model that will be used to do inference.

        var : str
          The kinematic variable to update. It cah have the values: 'R', 'V', 'A' or
          'RV'. The value 'RV' corresponds to updating simultaneously to 'R' and 'V'.

        new_value : numpy.ndarray or float
          The new value(s) to assign and substitute the old one(s)

        when : int or list
          The timesteps to update
        """

        if var == 'RV':
            self.RVA[who][when, :, 0:4] = new_value
        if var == 'R':
            self.RVA[who][when, :, 0:2] = new_value
        if var == 'V':
            self.RVA[who][when, :, 2:4] = new_value
        if var == 'A':
            self.RVA[who][when, :, 4:6] = new_value

    def get_value(self, who, var, when=None):
        """Returns values of some elements of RVA

        Parameters
        ----------
        who : str
          The element of the dictionary to update. If 'Truth' it refers to the
          matrix of values generated intregraing the equations of motion and it is
          the ground truth. Otherwise it must have the name of the machine lerarning
          model that will be used to do inference.

        var : str
          The kinematic variable to update. It cah have the values: 'R', 'V', 'A' or
          'RV'. The value 'RV' corresponds to updating simultaneously to 'R' and 'V'.

        when : int or list or None, optional
          The timesteps to update. If None it will consider all timesteps.

        Returns
        ----------
        numpy.ndarray
          The kinematic values
        """

        if when is None:
            when = np.arange(self.t_steps)
        if var == 'RV':
            return self.RVA[who][when, :, 0:4]
        if var == 'R':
            return self.RVA[who][when, :, 0:2]
        if var == 'V':
            return self.RVA[who][when, :, 2:4]
        if var == 'A':
            return self.RVA[who][when, :, 4:6]

    def get_target(self, var):
        """Returns the target matrix from the ground truth

        Parameters
        ----------
        var : str
          The kinematic variable that will be the target to return. It cah have the
          values: 'R', 'V' or 'A'

        Returns
        ----------
        numpy.ndarray
          The kinematic values that will be used as target
        """

        if var == 'R':
            return self.RVA['truth'][1:, :, 0:2]
        if var == 'V':
            return self.RVA['truth'][1:, :, 2:4]
        if var == 'A':
            return self.RVA['truth'][1:, :, 4:6]

    def get_features(self):
        """Returns the feature matrix the from ground truth

        Returns
        ----------
        numpy.ndarray
          The features matrix
        """

        return self.get_value(who='truth', var='RV')

    def simulate(self, interaction, t_steps, **kwargs):
        """Computes the configurations of the particles/agents according to the
        interaction selected using a numerical method at different timesteps by
        integrating the equations of motion

        Parameters
        ----------
        interaction : function
          Function that returns the acceleration of the particles according to their
          interaction

        t_steps : int
          Number of timesteps of the simulation
        """

        t_steps0 = self.t_steps
        RV0 = self.RVA['truth']

        self.t_steps = self.t_steps + t_steps
        self.RVA['truth'] = np.zeros((self.t_steps + 1, self.n_body, 6))
        self.RVA['truth'][0: t_steps0 + 1] = RV0

        for t_i in range(t_steps0, self.t_steps):
            R = self.get_value(who='truth', var='R', when=t_i)
            V = self.get_value(who='truth', var='V', when=t_i)
            A_new = interaction(R, V, dt=self.dt, **kwargs)
            V_new = V + A_new * self.dt
            R_new = R + V_new * self.dt
            self.update_value(who='truth', var='R', new_value=R_new, when=t_i + 1)
            self.update_value(who='truth', var='V', new_value=V_new, when=t_i + 1)
            self.update_value(who='truth', var='A', new_value=A_new, when=t_i + 1)

    def predict_instant(self, model, who, t):
        """Computes a new configuration of the particles/agents using a machine
        learning model for a timestep

        Parameters
        ----------
        model : model
          Model object capable of predicting

        who : str
          The element of the dictionary that is going to use as an input for the
          model. If 'Truth' it refers to the matrix of values generated intregraing
          the equations of motion and it is the ground truth. Otherwise it must have
          the name of the machine lerarning model that will be used to do inference.

        t : int
          Timestep that will be used as input

        Returns
        ----------
        tuple
          Tuple of numpy.ndarrays that represent the new values of the kinematic
          variables that have been predicted using the model. The number of values
          provided depend on the kind of model used.
        """

        if model.output_var == 'R':
            pass
        if model.output_var == 'V':
            x = self.get_value(who=who, var='RV', when=[t])
            V_new = model.predict(x)
            R_new = self.get_value(who=who, var='R', when=[t]) + V_new * self.dt
            return R_new, V_new
        if model.output_var == 'A':
            x = self.get_value(who=who, var='RV', when=[t])
            A_new = model.predict(x)
            V_new = self.get_value(who=who, var='V', when=[t]) + A_new * self.dt
            R_new = self.get_value(who=who, var='R', when=[t]) + V_new * self.dt
            return R_new, V_new, A_new

    def compute_mse(self, who1, who2, t):
        """ Computes the Mean Squared Error (MSE) of the position between to instant
        frames of two sources

        Parameters
        ----------
        who1 : str
          The element of the dictionary that is going to use to compute the MSE. If
          'Truth' it refers to the matrix of values generated intregraing the
          equations of motion and it is the ground truth. Otherwise it must have the
          name of the machine lerarning model that will be used to do inference.

        who2 : str
          The element of the dictionary that is going to use to compute the MSE. If
          'Truth' it refers to the matrix of values generated intregraing the
          equations of motion and it is the ground truth. Otherwise it must have the
          name of the machine lerarning model that will be used to do inference.

        Returns
        ----------
        float
        """

        r1 = self.get_value(who=who1, var='R', when=[t])
        r2 = self.get_value(who=who2, var='R', when=[t])

        return mse(r1, r2)

    def predict_trajectory(self, model):
        """Computes a new configuration of the particles/agents using a machine
         learning model (or multiple) for several timesteps and the MSE with respect
         to the ground truth

        Parameters
        ----------
        model : BaseModel object or list of BaseModel objects
          Model or models capable of predicting
        """

        if type(model) != list:
            model = [model]
        for model in model:
            self.RVA[model._name] = np.zeros((self.t_steps + 1, self.n_body, 6))
            self.update_value(who=model._name, var='RV', new_value=self.get_value(who='truth', var='RV', when=[0]),
                              when=[0])
            for t in range(self.t_steps):
                if model.output_var == 'R':
                    pass
                if model.output_var == 'V':
                    R_new, V_new = self.predict_instant(model=model, who=model._name, t=t)
                    self.update_value(who=model._name, var='R', new_value=R_new, when=[t + 1])
                    self.update_value(who=model._name, var='V', new_value=V_new, when=[t + 1])
                if model.output_var == 'A':
                    R_new, V_new, A_new = self.predict_instant(model=model, who=model._name, t=t)
                    self.update_value(who=model._name, var='R', new_value=R_new, when=[t + 1])
                    self.update_value(who=model._name, var='V', new_value=V_new, when=[t + 1])
                    self.update_value(who=model._name, var='A', new_value=V_new, when=[t + 1])

            self.MSE[model._name] = [self.compute_mse(who1='truth', who2=model._name, t=t) for t in
                                     range(self.t_steps + 1)]

    def plot_instant(self, t, who, ax, **kwargs):
        """Plots the configuration of the system at a selected timestep

        Parameters
        ----------
        t : int
          Timestep to plot

        who : str
          The element of the dictionary to be plotted. If 'Truth' it refers to the
          matrix of values generated intregraing the equations of motion and it is
          the ground truth. Otherwise it must have the name of the machine lerarning
          model that will be used to do inference.

        ax : numpy.ndarray
          Contains matplotlib.axes._subplots.AxesSubplot objects
        """
        if self.n_body <= 10:
            colors = mcolors.TABLEAU_COLORS
            # colors = mcolors.BASE_COLORS.copy()
            # colors.pop('w',None)
            # colors.pop('k',None)
        else:
            colors = mcolors.CSS4_COLORS
        c = [color for color, _ in zip(colors, range(self.n_body))]
        R = self.get_value(who=who, var='R', when=t)
        ax.scatter(R[:, 0], R[:, 1], c=kwargs.get('c', c), s=kwargs.get('s', 100), alpha=kwargs.get('alpha', 1),
                   edgecolors=kwargs.get('edgecolors', c), linewidths=kwargs.get('linewidths'),
                   marker=kwargs.get('marker', None), label=kwargs.get('label', None))

    def plot_step_comparison(self, model, ax, t=0, **kwargs):
        """Plots a comparison between the calculations done using a numerical method
        and the results obtained by inference using a machine learning model at one
        timestep

        Parameters
        ----------
        model : model
          Model object capable of predicting

        ax : numpy.ndarray
          Contains matplotlib.axes._subplots.AxesSubplot objects

        t : int, optional
          Timestep to plot
        """

        kwargs2 = kwargs.copy()
        kwargs2['c'] = 'white'
        if 'c' in kwargs.keys():
            kwargs2['edgecolors'] = kwargs['c']
        self.plot_instant(t=t, who='truth', ax=ax, alpha=1, linewidths=3, **kwargs2, label='Start Truth')
        self.plot_instant(t=t + 1, who='truth', ax=ax, edgecolors='black', marker='o', s=100, linewidths=2, **kwargs,
                          label='End Truth')
        New = self.predict_instant(model, who='truth', t=t)
        R_new = New[0]
        self.RVA['single_prediction_' + model._name] = R_new
        self.plot_instant(t=0, who='single_prediction_' + model._name, ax=ax, edgecolors='black', marker='D', s=100,
                          linewidths=2, **kwargs, label='End Prediction')

    def plot_trajectory(self, who, ax, **kwargs):
        """Plots the configuration of the system at the diferent times to show the
        trajectories of the agents/particles

        Parameters
        ----------
        who : str
          The element of the dictionary to be plotted. If 'Truth' it refers to the
          matrix of values generated intregraing the equations of motion and it is
          the ground truth. Otherwise it must have the name of the machine lerarning
          model that will be used to do inference.

        ax : numpy.ndarray
          Contains matplotlib.axes._subplots.AxesSubplot objects
        """

        for t in range(self.t_steps):
            self.plot_instant(t=t, who=who, ax=ax, alpha=0.1, **kwargs)
        kwargs2 = kwargs.copy()
        kwargs2['c'] = 'white'
        if 'c' in kwargs.keys():
            kwargs2['edgecolors'] = kwargs['c']
        self.plot_instant(t=0, who=who, ax=ax, alpha=1, linewidths=3, **kwargs2, label='Start')
        self.plot_instant(t=self.t_steps, who=who, ax=ax, edgecolors='black', marker='o', s=100, linewidths=2, **kwargs,
                          label='End')

    def plot_trajectory_tex(self, who, ax):
        """Plots on an ax the trajectories of all the agents/particles with a format
        suitable for the library tikzplotlib
        Parameters
        ----------
        who : str
          The element of the dictionary to be plotted. If 'Truth' it refers to the
          matrix of values generated intregraing the equations of motion and it is
          the ground truth. Otherwise it must have the name of the machine lerarning
          model that will be used to do inference.

        ax : numpy.ndarray
          Contains matplotlib.axes._subplots.AxesSubplot objects
        """
        if self.n_body <= 6:
            colors = mcolors.TABLEAU_COLORS
        else:
            colors = mcolors.CSS4_COLORS
        c = [color for color, _ in zip(colors, range(self.n_body))]

        for i, color in zip(range(self.n_body), colors):
            x = self.RVA[who][:, i, 0:1]
            y = self.RVA[who][:, i, 1:2]
            ax.scatter(x, y, alpha=0.3, c=color, s=100, label=f'body {i}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.tick_params(direction='in')
        ax.tick_params(labelbottom=False, labelleft=False, right=True, top=True)
        # ax.legend()

    def plot_trajectory_comparison(self, model, ax):
        """Plots a side-by-side comparison between the calculations done using a
        numerical method and the results obtained by inference using a machine
        learning model for a trajectory

        Parameters
        ----------
        model : model
          Model object capable of predicting

        ax : numpy.ndarray
          Contains matplotlib.axes._subplots.AxesSubplot objects
        """

        # self.predict_trajectory(model)
        self.plot_trajectory('truth', ax=ax[0])
        self.plot_trajectory(model._name, ax=ax[1])

    def trajectory_to_gif(self, who, filename, fps=1, **kwargs):
        """Creates a gif file of the different configurations of the system at
        different timesteps

        Parameters
        ----------
        who : str
          The element of the dictionary to be plotted. If 'Truth' it refers to the
          matrix of values generated intregraing the equations of motion and it is
          the ground truth. Otherwise it must have the name of the machine lerarning
          model that will be used to do inference.

        filename : str
          Name of the file without the extension. The extension '.gif' is added

        fps : int, optional
          The number of frames per second of the gif( or images per second)
          displayed
        """

        img_list = []
        for t in range(self.t_steps):
            fig, ax = plt.subplots()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            self.plot_instant(t=t, who=who, ax=ax, alpha=0.5, **kwargs)
            img_list.append(fig_to_img(fig))
            plt.close(fig)
        gif_from_list(img_list, filename, fps=fps)


