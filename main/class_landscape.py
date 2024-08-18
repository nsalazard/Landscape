import numpy as np
import random
from copy import deepcopy, copy

from morphogen_regimes import mr_sigmoid
from class_module import Node

class Landscape:
    def __init__(self,cell, module_list=(), A0=0., init_cond=(0., 1.), regime=mr_sigmoid, n_regimes=2,
                 morphogen_times=(0.,), used_fp_types=(Node,), immutable_pars_list=(), tiltx = 0,tilty = 0, tilt_par = None, xy = False):
        """
        :param module_list: list of module objects
        :param A0: float - strength of global attraction (boundary condition of the potential)
        :param init_cond: default initial condition for cells in this landscape
        :param regime: function from morphogen_regimes - dynamics of module amplitudes and sizes
        :param n_regimes: number of morphogen conditions
        :param morphogen_times: times of changes in morphogens (signals)
        :param used_fp_types: list of module types to add in random mutation (Node, UnstableNode, Center, NegCenter)
        :param immutable_pars_list: list of parameter names to fix for all initial modules of the landscape
        """
        self.module_list = []
        # modules are stored in module_list
        for ind in range(len(module_list)):
            self.module_list.append(deepcopy(module_list[ind]))
            for par_name in immutable_pars_list:
                if par_name in self.module_list[ind].mutable_parameters_list:
                    self.module_list[ind].remove_mutable_parameter(par_name)
        self.cell = cell
        self.A0 = A0
        self.regime = regime
        self.n_regimes = n_regimes
        self.morphogen_times = morphogen_times
        self.used_fp_types = used_fp_types
        self.init_cond = init_cond
        self.max_n_modules = 7  #MAX
        self.tiltx = tiltx
        self.tilty = tilty
        self.tilt_var_x = tiltx
        self.tilt_var_y = tilty
        self.tilt_par = tilt_par
        self.num_dim = 2
        self.xy = xy

        count = 0
        for module in module_list:
            if isinstance(module, Node):
                count += 1
        self.cell.n_attrac = count

        self.fitness = None  # stores calculated fitness
        self.result = None  # stores some ouput besides fitness

        self.cell_coordinates = None  # current coordinates of cells
        self.cell_states = None  # current cell state assignments
        self.trajectories = None

    def __repr__(self):
        if not self.module_list:
            return 'Empty landscape'
        repr_str = 'Landscape with modules:'
        for module in self.module_list:
            module_str = module.__str__()
            repr_str += '\n' + module_str + ','
        repr_str = repr_str[:-1]
        return repr_str
#______________________________________________________________________________________________________________

    def ModifyTilt_x(self,t,k=0.5):
        k= self.tilt_par
        tilt_new = self.tiltx - self.tiltx * (1 / (1 + np.exp(-k * (t - (self.morphogen_times[0])))))
        self.tilt_var_x = round(tilt_new, 6)

    def ModifyTilt_y(self,t,k=0.5):
        k= self.tilt_par
        tilt_new = self.tilty - self.tilty * (1 / (1 + np.exp(-k * (t - self.morphogen_times[0]))))
        self.tilt_var_y = round(tilt_new, 6)

    def ModifyTilt_xy(self,t,k=0.5, times = None):
        if times is None:
            times = (int(self.morphogen_times[0]), int((self.morphogen_times[0] + self.cell.tf)/2))
        time1= int((self.cell.t0 + times[0])/2	)
        time2= int((times[0] + times[1])/2	)
        time3= int((times[1] + self.cell.tf)/2	)

        k= self.tilt_par
        
        if t < time1:

            tilt_new = self.tiltx * (1 / (1 + np.exp(-k * (time1 - t))))
            self.tilt_var_x = round(tilt_new, 6)
            self.tilt_var_y = 0
        elif time1 <= t < time2:

            tilt_new = self.tilty * (1 / (1 + np.exp(-k * (t- time2))))
            self.tilt_var_x = 0
            self.tilt_var_y = round(tilt_new, 6)
        else:

            tilt_new = self.tilty * (1 / (1 + np.exp(-k * (time3 - t))))
            self.tilt_var_x = 0
            self.tilt_var_y = round(tilt_new, 6)

# _______________________________________________________________________________________________________________
# ____________________________ Landscape dynamics calculation____________________________________________________

    @staticmethod
    def local_weight(r, sig):
        """ Potential kernel (gaussian) """
        weight = np.exp(-0.5 * (r / sig) ** 2)
        return weight

    @staticmethod
    def fixed_point(module, x, y):
        J = module.J  # Jacobian of the module
        dx = J[0][0] * x + J[0][1] * y
        dy = J[1][0] * x + J[1][1] * y
        return dx, dy

    def __call__(self, t, q,t0=0,return_potentials=False):
        """
        Evaluate the flow at coordinates q and time t
        :param t: float
        :param q: array of shape (2, m, n); q[0] are x-coordinates, q[1] are y-coordinates
        :param return_potentials: bool
        :return: tuple of arrays with x and y derivatives, potentials (optional)
        """
        if self.tilt_par != None and self.xy == False:
            self.ModifyTilt_x(t)
            self.ModifyTilt_y(t)

        elif self.tilt_par != None and self.xy == True:
            self.ModifyTilt_xy(t)

        if (self.num_dim ==1):
          x = q[0]
          w = np.zeros((len(self.module_list), *x.shape))
          dx = np.zeros((len(self.module_list), *x.shape))
          derivs = self.A0 * -x ** 3 + np.sum(w * dx, axis=0) - self.tilt_var_x
          if return_potentials:
              potential = self.A0 / 4 * (x ** 4) + self.tilt_var_x*x
              rot_potential = 0.
              return derivs, potential, rot_potential
          return derivs

        else:
          x = q[0]
          y = q[1]
          w = np.zeros((len(self.module_list), *x.shape))
          sig = np.zeros((len(self.module_list)))
          sign = np.zeros((len(self.module_list)), dtype='int')   # Sign of modules (+1 or -1)
          curl = np.zeros((len(self.module_list)), dtype='bool')  # Is the module rotational (0 or 1)
          dx, dy = np.zeros((len(self.module_list), *x.shape)), np.zeros((len(self.module_list), *x.shape))
          for i, module in enumerate(self.module_list):
              V, sig[i], A = module.get_current_pars(t, self.regime, *self.morphogen_times) #here
              if module.__class__.__name__ == 'Node' or module.__class__.__name__ == 'NegCenter':
                  sign[i] = -1
              else:
                  sign[i] = +1
              if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
                  curl[i] = 1

              xr = x - module.x
              yr = y - module.y
              r = np.sqrt(xr ** 2 + yr ** 2)
              w[i, :] = A * self.local_weight(r, sig[i])
              dx[i, :], dy[i, :] = self.fixed_point(module, xr, yr)
          derivs = self.A0 * np.array((-x ** 3, -y ** 3)) + (np.sum(w * dx, axis=0), np.sum(w * dy, axis=0)) - (np.full(x.shape,self.tilt_var_x),np.full(y.shape,self.tilt_var_y))
          if return_potentials:

              potential = np.sum(w * (~curl * sign * sig ** 2)[:, np.newaxis, np.newaxis], axis=0) + self.A0 / 4 * (x ** 4 + y ** 4) + self.tilt_var_x*x + self.tilt_var_y*y
              rot_potential = np.sum(w * (curl * sign * sig ** 2)[:, np.newaxis, np.newaxis], axis=0)
              return derivs, potential, rot_potential
          return derivs
#_____________________________________________________________________________________________________________
#___________________________________ For evolutionary optimization____________________________________________

    def get_fitness(self, fitness_pars):
        """ Compute the fitness function, using a dict of fitness parameters. Override in child classes. """
        raise NotImplementedError

    def add_module(self, M):
        """ Add a module to the landscape """
        self.module_list.append(deepcopy(M))

    def del_module(self, del_ind):
        """ Remove the module at index del_ind from the landscape """
        del self.module_list[del_ind]

    def mutate(self, par_limits, par_choice_values, prob_pars, fitness_pars):
        """
        In-place modification of the landscape.
        Randonly mutate the landscape according to prob_pars. If a parameter is mutated, new values are sampled
        according to par_limits and par_choice_values. Recalculate the fitness using fitness_pars.
        :param par_limits:
        :param par_choice_values:
        :param prob_pars:
        :param fitness_pars:
        """
        r = np.random.uniform()

        if r < prob_pars['prob_tilt']/2:
            self.tiltx = np.random.uniform(*self.par_choice_values['tilt_lmt'])
        
        elif r < prob_pars['prob_tilt']:
            self.tiltx = np.random.uniform(*self.par_choice_values['tilt_lmt'])

        elif (prob_pars['prob_tilt'] <= r < prob_pars['prob_tilt'] + prob_pars['prob_add']) and len(self.module_list) < self.max_n_modules:
            # print('Adding,', 'len =', len(self.module_list), ', r =', r)
            fp_type = random.choice(self.used_fp_types)
            self.add_module(fp_type.generate(par_limits, par_choice_values, n_regimes=self.n_regimes))
        elif r < (prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_tilt']) and len(self.module_list) > 1:
            # print('Deleting,', 'len =', len(self.module_list), ', r =', r)
            del_ind = np.random.choice(len(self.module_list))
            self.del_module(del_ind)
        elif r < (prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] + prob_pars['prob_tilt']) and len(
                self.module_list) > 1:
            # print('Shuffling,', 'len =', len(self.module_list), ', r =', r)
            random.shuffle(self.module_list)
        else:
            # print('Modifying,', ', r =', r)
            mod_ind = np.random.choice(len(self.module_list))
            self.module_list[mod_ind].mutate(par_limits, par_choice_values)
        self.fitness = self.get_fitness(fitness_pars)

    # MARK: - number_attractors   

    def number_attractors(self):
        count = 0
        for module in self.module_list:
            if isinstance(module, Node):
                count += 1     
        self.cell.n_attrac = count

    def mutate_and_return(self, par_limits, par_choice_values, prob_pars, fitness_pars):
        """
        Mutates and also returns the landscape - required for parallel computation.
        Randonly mutate the landscape according to prob_pars. If a parameter is mutated, new values are sampled
        according to par_limits and par_choice_values. Recalculate the fitness using fitness_pars.
        :param par_limits:
        :param par_choice_values:
        :param prob_pars:
        :param fitness_pars:
        """
        ti = np.random.uniform()
        r = np.random.uniform()
        if (r < prob_pars['prob_add'] or len(self.module_list) == 0) and len(self.module_list) < self.max_n_modules:
            fp_type = random.choice(self.used_fp_types)
            self.add_module(fp_type.generate(par_limits, par_choice_values, n_regimes=self.n_regimes))
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] and len(self.module_list) > 1 \
                or len(self.module_list) > self.max_n_modules:
            del_ind = np.random.choice(len(self.module_list))
            self.del_module(del_ind)
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] and len(
                self.module_list) > 1:
            random.shuffle(self.module_list)
        else:
            mod_ind = np.random.choice(len(self.module_list))
            self.module_list[mod_ind].mutate(par_limits, par_choice_values)
        self.fitness = self.get_fitness(fitness_pars)
        return self

# ______________________________________________________________________________________________________________________
# _____________________________________ Everything to do with cells _______________________________________________

# MARK: - init_cells

    def init_cells(self, n, init_cond, noise=0.):
        """
        Initialize cells in the landscape with a given initial condition.
        :param n: int, number of cells
        :param init_cond: int or array/tuple of length 2 or array of shape (2, n) or array of length self.module_list.
            Int: module number, all cells are initialized at the module location.
            Tuple: (x,y) - same coordinate for all cells.
            Array (2, n) - x and y coordinates for n cells.
            Array (len(self.module_list)) - number of cells starting at each module, numbers must sum to n.
        :param noise: amplitude of gaussian noise added to each cell's initial coordinate.
        """
        if isinstance(init_cond, int):
            module0 = self.module_list[init_cond]
            init_cond = (module0.x, module0.y)
        elif init_cond is None:
            init_cond = self.init_cond

        init_cond = np.asarray(init_cond)
        if init_cond.shape == (2, n):
            self.cell_coordinates = init_cond.astype('float')
        elif init_cond.shape == (2,):
            self.cell_coordinates = np.tile(init_cond.astype('float'), (n, 1)).T
        elif len(init_cond) == len(self.module_list) and np.sum(init_cond) == n:
            module_locs = np.array([(module.x, module.y) for module in self.module_list])
            self.cell_coordinates = np.repeat(module_locs, init_cond, axis=0).T
        else:
            print('Wrong shape of init_cond input')
            self.cell_coordinates = np.ones((2, n)) * np.nan

        if noise != 0.:
            self.cell_coordinates += noise * np.random.randn(2, n)
        self.cell_states = self.get_cell_states_static()

    def reset_cells(self):
        """ Remove all stored cell coordinates and states """
        self.cell_coordinates = None
        self.cell_states = None

    @property
    def n(self):
        """ Number of cells currently in the landscape """
        return np.sum(~np.isnan(np.sum(self.cell_coordinates, axis=0)))
    
# MARK: - get_cell_states_static  

    def get_cell_states_static(self, coordinate=None):
        """
        Return cell states given cell coordinates.
        Assignment is based on proximity to modules and does not depend on time or signals.
        :param coordinate: array of shape (2, n) where n is the number of cells
            (optional, can use the current coordinates stored in landscape)
        :return: states - array of length n of ints
        """
        if coordinate is None:
            coordinate = self.cell_coordinates
        # TODO: take into account only node modules
        dist = np.empty((coordinate.shape[1], len(self.module_list)))
        for i, module in enumerate(self.module_list):
            dist[:, i] = np.linalg.norm(coordinate.T - np.array((module.x, module.y)), axis=1)
        states = np.argmin(dist, axis=1)

        return states

# MARK: - get_cell_states

    def get_cell_states(self, t, coordinate=None, measure='gaussian'):
        """
        Return cell states given cell coordinates. Assignent based on a chosen distance measure, can depend on time or signals.
        :param t: float, timepoint
        :param coordinate: array of shape (2, n) where n is the number of cells
            (optional, can use the current coordinates stored in landscape)
        :param measure: 'dist' - base on Euclidean distance to modules, same as get_cell_states_static.
            'gaussian' - based on a gaussian mixture model, taking into account time-dependent module size.
        :return: states - array of length n of ints
        """
        if coordinate is None:
            coordinate = self.cell.Positions[:, :, t] #self.cell_coordinates
        states = None

        if measure == 'dist':
            dist = np.empty((coordinate.shape[1], len(self.module_list)))
            for i, module in enumerate(self.module_list):
                dist[:, i] = np.linalg.norm(coordinate.T - np.array((module.x, module.y)), axis=1)
            states = np.argmin(dist, axis=1)
        elif measure == 'gaussian':
            prob = np.zeros((coordinate.shape[1], len(self.module_list) + 1))
            for i, module in enumerate(self.module_list):
                V, st, at = module.get_current_pars(t, self.regime, *self.morphogen_times)
                prob[:,i+1] = np.exp(
                    -np.sum((coordinate.T - np.array((module.x, module.y))) ** 2, axis=1) / 2. / st ** 2) / st ** 2
            # print(prob/2/np.pi)
            #prob = (prob.T / np.sum(prob, axis=1)).T
            prob[:, 0] = 0.1 # probability threshold: for probs below this value cells will be assigned as 'unclustered'
            # print(prob*100)
            states = np.argmax(prob, axis=1)
        return states

# MARK: - run_cells

    def run_cells(self, noise=0., frozen=False, t_freeze=None, same_time=True, measure='gaussian', ndt=100):
        """
        Run trajectories for cells in the landscape.
        :param t0: float, start time
        :param tf: float, end time
        :param nt: int, number of timepoints
        :param noise: float, amplitude of gaussian noise
        :param ndt: int, number of integration steps per timepoint
        :param frozen: bool, whether to fix the landscape paremeters
        :param t_freeze: if frozen, provide the time at which to calculate the landscape, to be kept constant
        :return: traj (array of shape (2, n, nt)) and states (int array of shape (2, nt))
        """
        #traj = np.empty((*self.cell_coordinates.shape, nt), dtype='float')
        #states = np.empty((self.cell_coordinates.shape[1], nt), dtype='int')

        t0, tf, nt = self.cell.t0, self.cell.tf, self.cell.nt
        t = t0
        aux = self.cell.pos.copy()
        n_data = nt / self.cell.num_data
        ii = 0

        self.cell.States[:, 0] = self.get_cell_states(0, measure= measure, coordinate=aux)
        Delta_t = (tf - t0) / (nt)
        dt = Delta_t / ndt
        sqrt_dt = np.sqrt(dt)

        if frozen:
            def f(t, q):
                return self(t_freeze, q)
        else:
            f = self
        if same_time:
            for Delta_step in range(1, nt):
                for dt_step in range(ndt):
                    aux += f(t, aux) * dt + noise * np.random.standard_normal(aux.shape) * sqrt_dt
                    t += dt
                self.cell.Positions[:, :, Delta_step] = aux
                self.cell.States[:, Delta_step] = self.get_cell_states(Delta_step, coordinate=aux)

        else:
            tt=0
            #(tf-t0), just the diff matters
            delta = (nt)/(tf - t0)
            idx_max= 0
            lst = [int(self.cell.intvl*delta * x) for x in np.arange(self.cell.div)]
            while tt < nt:
                if tt in lst:
                    idx_max += self.cell.repl
                for dt_step in range(ndt):
                    aux[:,:idx_max] += f(t, aux[:,:idx_max]) * dt + noise * np.random.standard_normal(aux[:,:idx_max].shape) * sqrt_dt
                    t += dt
                self.cell.Positions[:, :idx_max, tt] = aux[:,:idx_max]
                self.cell.States[:, tt] = self.get_cell_states(tt, coordinate=aux)
                tt += 1
        self.cell_states = self.cell.States[:, -1]
