import numpy as np

class Cells:
    def __init__(self, t0, tf,tc, div=1, repl=1, nt=100, init_cond=(0, 0), num_data = 10, W_H_d = None, W_H_dp = None):
        self.t0 = t0
        self.tf = tf
        self.tc = tc
        self.div = div
        self.repl = repl
        self.nt = nt
        self.init_cond = init_cond
        self.num_cells = div * repl
        self.intvl = (tc - t0) / div
        self.n_attrac = 0
        self.h_diversity = 0
        self.h_div_pos = 0
        self.h_px = 0
        self.final_entropy = 0
        self.num_data = num_data
        self.W_H_d = W_H_d
        self.W_H_dp = W_H_dp
        self.penal = 0
        # Arrays
        self.Start_Times = np.zeros(self.num_cells, dtype='int')
        self.End_Times = np.empty(self.num_cells, dtype='float')
        self.Positions = np.empty((2, self.num_cells, self.nt), dtype='float')
        self.mtx_div_time = np.empty((self.div, self.nt), dtype='int')
        self.pos = None
        self.States = np.zeros((self.num_cells, nt), dtype='int')
        self.prob_attrac = None
        self.prob_ts = None
        self.mtx_prob_ts = None
        self.data_States = None


    def init_position(self, noise=0.5):
        init = np.asarray(self.init_cond).astype(float)
        self.pos = np.empty((2, self.num_cells), dtype='float')
        self.pos = np.tile(init, (self.num_cells, 1)).T
        if noise != 0.:
            self.pos += noise * np.random.randn(2, self.num_cells)
        # Make a writable copy instead of broadcasting
        for ii in range(self.nt):
            self.Positions[:, :, ii] = self.pos.copy()

    # Function to create an array of cells
    def create_Start_Times(self):
        t_start = self.t0
        index = 0
        for ii in range(self.div):
            for jj in range(self.repl):  # t_start is the same for all the replicates
                if index < self.num_cells:
                    self.Start_Times[index] = t_start
                    index += 1
            t_start += self.intvl

    def Prob_Atrrac(self):
        #self.prob_attrac = np.bincount(self.States[:,-1], minlength=self.n_attrac)
        self.prob_attrac = np.bincount(self.States[:,-1])
        #self.n_attrac = len(self.prob_attrac)

    def Prob_ts(self):
        ci = 0
        cf = self.repl
        lst = []
        for kk in range(self.div):
          lst.append(np.bincount(self.States[ci:cf, -1],minlength=self.n_attrac)) # [7,9,1]
          ci += self.repl
          cf += self.repl
        self.prob_ts = np.array(lst[:np.prod((self.div,self.n_attrac))]).reshape((self.div,self.n_attrac))
        self.mtx_prob_ts = np.empty((self.div,self.n_attrac), dtype=float)

    def H_diver(self):
        entropy = 0.0
        for count in self.prob_attrac:
            if count > 0:
                probability = count / self.num_cells
                entropy += probability * np.log2(probability)
        self.h_diversity = -1 * entropy

    def H_div_pos(self):
        entropy = 0.0
        for ii in range(self.div):
            for jj in range(self.n_attrac):
                count = self.prob_ts[ii,jj]
                if count <= 0:
                    self.mtx_prob_ts[ii,jj] = 0.
                elif count > 0:
                    probability = count / self.repl
                    self.mtx_prob_ts[ii,jj] =  probability
                    entropy += probability * np.log2(probability)
                    #print(f'{self.mtx_prob_ts[ii,jj]}  {probability}  {probability * np.log2(probability)}')
        self.h_div_pos = (-1./self.div) * entropy

    def H_px(self):
        self.h_px = -1*np.log2(1/self.div) # L * (1/L) * log(1/L)

    def compare_states(self):
        # Extract the two slices to be compared
        x = int ((self.tf - self.tc)/2)

        slice_1 = self.states[:, :, -1]
        slice_2 = self.states[:, :, -x]
        
        # Compare the slices and count the differing elements
        differences = slice_1 != slice_2
        counter = np.sum(differences)
        self.penal = counter

    def Entropy(self):
        if self.W_H_dp is None and self.W_H_d is None:
            self.W_H_dp = 1
            self.W_H_d = 1
        self.final_entropy = - (self.W_H_d * self.h_diversity) + ((self.W_H_dp * self.h_div_pos) - self.h_px)

    def get_data_concentration(self):
        n_data = self.nt / self.num_data
        self.data_States = np.zeros((self.num_data,self.n_attrac), dtype='int')
        for ii in range(self.num_data):
            self.data_States[ii,:] = np.bincount(self.States[:,int(ii*n_data)], minlength=self.n_attrac)
        data_States = self.data_States / self.num_cells
        self.data_States = np.round(data_States, 2)

    def fig_div_time(self):
        for ii in range (self.nt):
            ci = 0
            cf = self.repl
            lst = []
            for kk in range(self.div):
              lst.append(np.bincount(self.States[ci:cf,ii],minlength=self.n_attrac)) # [7,9,1]
              ci += self.repl
              cf += self.repl
            mtx_prob_att = np.array(lst[:np.prod((self.div,self.n_attrac))]).reshape((self.div,self.n_attrac))
            mtx_max_prob = np.argmax(mtx_prob_att, axis=1)
            #print("Shape of mtx_max_prob:", mtx_max_prob.shape)
            #print(mtx_max_prob)
            self.mtx_div_time [:,ii] = mtx_max_prob

