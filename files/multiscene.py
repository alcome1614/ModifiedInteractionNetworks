from scene import *

################################################################################
#                                   MULTISCENE                                 #
################################################################################
class MultiScene():
    """
    A class that contains multiple Scene objects

    ...

    Attributes
    ----------
    scene_list : list
        The list of Scene objects

    : int
      The number of Scenes in the list

    t_steps : int
      The maximum number of steps out of the Scenes in the list

    MSE : dict
      Contains different entries each of them contains time lists of value of the
      Mean Squared Error (MSE)  of the position (R) of theparticles/agents at
      every instant of time with respect to the 'truth' entry which corresponds to
      the ground truth. T. The entries have the name of the model used to predict
      the next timesteps.

    Methods
    -------
    add_scene(self, scene)
      Adds a Scene object to the list

    add_bunch(self, , n_body, t_steps, R_init, V_init, interaction, r=0.2, v=0.1, dt=1., **kwargs)
      Adds multiple Scene objects to the list with certain parameters

    get_features(self, joint=False)
      Returns the feature matrix(ces) of the Scenes in the list

    get_target(self, var, joint=False)
      Returns the target matrix of the Scenes in the list

    __call__(self, scene_index)
      Returns one element of the list of Scenes

    predict_trajectory(self, model)
      Computes a new configuration of the particles/agents using a machine
      learning model (or multiple) for several timesteps and the MSE with respect
      to the ground truth for each scene individually

    compute_mse(self)
      Computes the Mean Squared Error (MSE) of the position for the different
      models from their MSE combined

    plot_mse(self, ax)
      Plots a comparison of the mse in the position for different mulstistep
      rollout lengths

    def save(self, filename)
      Saves the dictionary with all attributes in a file using pickle

    def load(self, filename)
      Loads the  dictionary with all attributes from a file using pickle
    """

    def __init__(self, load=None):
        """
        Parameters
        ----------
        load : str or None, optional
          Name of the file from which to load the attributes
        """

        if load:
            self.load(load)
        else:
            self.scene_list = []
            self.n_scenes = 0
            self.MSE = {}

    def add_scene(self, scene):
        """Adds a Scene object to the list

        Parameters
        ----------
        scene : Scene object
          A Scene object
        """

        self.scene_list.append(scene)
        self.n_scenes = self.n_scenes + 1
        return

    def add_bunch(self, n_scenes, n_body, t_steps, R_init, V_init, interaction, r=0.2, v=0.1, dt=1., **kwargs):
        """Adds multiple Scene objects to the list with certain parameters

        Parameters
        ----------
        n_scenes : int
          Number of Scenes

        n_body: int
          Number of agents/particles per Scene

        t_steps: int
          Number of timesteps per Scene

        R_init: func
          Function that returns and initializes the position matrix of the
          agents/particles

        V_init: func
          Function that returns and initializes the velocity matrix of the
          agents/particles

        interaction: func
          Function that returns the acceleration of the particles according to their
          interaction

        r : float, optional
          Maximum distance from the center of the particles generated

        v : float, optional
          Maximum norm of the velocities generated

        dt : int, optional
          Time increment for the simulations
        """

        for _ in range(n_scenes):
            escena = Scene(n_body, R_init, V_init, r=r, v=v, dt=dt)
            escena.simulate(interaction=interaction, t_steps=t_steps, **kwargs)
            self.add_scene(escena)

    def get_features(self, joint=False):
        """Returns the feature matrix(ces) of the Scenes in the list

        Parameters
        ----------
        joint : bool, optional
          If False returns a list of matrices else it returns a numpy.ndarray of
          matrices

        Returns
        -------
         list or numpy.ndarray
            Matrices with the features
        """

        if joint:
            return np.vstack([escena.get_features() for escena in self.scene_list])
        else:
            return [escena.get_features() for escena in self.scene_list]

    def get_target(self, var, joint=False):
        """Returns the target matrix of the Scenes in the list

        Parameters
        ----------
        var : str
          It returns the kinematic variable matrix selected. It can be: 'R', 'V',
          'A'

        joint : bool, optional
          If False returns a list of matrices else it returns a numpy.ndarray of
          matrices

        Returns
        -------
         list or numpy.ndarray
            Matrices with the targets
        """
        if joint:
            return np.vstack([escena.get_target(var) for escena in self.scene_list])
        else:
            return [escena.get_target(var) for escena in self.scene_list]

    def __call__(self, scene_index):
        """Returns one element of the list of Scenes

        Parameters
        ----------
        scene_index : int
          Index of the Scene in the list to be returned

        Returns
        -------
         scene
            Scene object
        """

        return self.scene_list[scene_index]

    def predict_trajectory(self, model):
        """Computes a new configuration of the particles/agents using a machine
        learning model (or multiple) for several timesteps and the MSE with respect
        to the ground truth for each scene individually

        Parameters
        ----------
        model : BaseModel object or list of BaseModel objects
          Model or models capable of predicting
        """

        for escena in self.scene_list:
            escena.predict_trajectory(model=model)

    def compute_mse(self):
        """Computes the Mean Squared Error (MSE) of the position for the different
        models from their MSE combined
        """

        self.MSE = {}
        self.t_steps = 0
        t_counter = np.zeros(self.n_scenes, dtype=int)
        for i, escena in enumerate(self.scene_list):
            t_counter[i] = escena.t_steps
            self.t_steps = max(self.t_steps, escena.t_steps)
        for model in self(0).MSE.keys():
            self.MSE[model] = np.zeros(self.t_steps + 1)
        body_counter = np.zeros(self.t_steps + 1, dtype=int)
        for t, escena in zip(t_counter, self.scene_list):
            body_counter[:t + 1] += escena.n_body
            for model in self(0).MSE.keys():
                self.MSE[model][:escena.t_steps + 1] += np.array(escena.MSE[model]) * escena.n_body
        for model in self(0).MSE.keys():
            self.MSE[model] /= body_counter

    def plot_mse(self, ax):
        """Plots a comparison of the mse in the position for different mulstistep
        rollout lengths

        Parameters
        ----------
        ax : numpy.ndarray
          Where the plot will be made
        """

        for model in self(0).MSE.keys():
            ax.plot([np.log10(x) for x in self.MSE[model][1:]], label=model)
        ax.set_title('Multistep Rollout MSE')
        ax.set_xlabel('Length of prediction (steps)')
        ax.set_ylabel('$\log_{10}$ MSE')
        # ax.legend()

    def save(self, filename):
        """Saves the dictionary with all attributes in a file using pickle

        Parameters
        ----------
        filename : str
          Name of the file without the extension. The extension '.pickle' is added
        """

        with open(f'{filename}.pickle', 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """Loads the  dictionary with all attributes from a file using
        pickle

        Parameters
        ----------
        filename : str
          Name of the file without the extension. The extension '.pickle' is added
        """

        with open(f'{filename}.pickle', 'rb') as handle:
            self.__dict__.update(pickle.load(handle))


