import numpy as np

from class_landscape import Landscape


class Somitogenesis_Landscape(Landscape):
    # fitness_pars: time_pars, init_state, ncells, t0_shift, noise, high_value, low_value, t_stable, penalty_weight

    def get_kymo(self, init_state, t0_shift=1., noise=0., ndt=100):
        kymo = np.zeros((self.cell.num_cells, self.cell.nt))
        for cell_ind in range(self.cell.num_cells):
            self.morphogen_times = (t0_shift * cell_ind,)
            #self.init_cells(1, init_state, noise=noise)
            self.cell.init_position(noise)
            self.run_cells(noise, ndt=ndt)   #HERE!
            kymo[cell_ind] = self.cell.Positions[0, 0, :]  # x-coordinate of the first (and only) cell in time
        self.morphogen_times = (self.cell.tc,)
        self.result = kymo
        return kymo

    def get_fitness(self, fitness_pars):
        #print("get_fitness-Segmentation")

        noise = fitness_pars['noise']
        ndt = fitness_pars['ndt']
        self.number_attractors()
        self.cell.init_position(noise)
        self.run_cells(noise, ndt=ndt, same_time=False, measure='gaussian')
        self.cell.Prob_Atrrac()
        self.cell.Prob_ts()
        self.cell.H_diver()
        self.cell.H_div_pos()
        #self.cell.H_px()
        kymo = self.cell.Positions[0, :, :]
        self.result = kymo
        self.cell.Entropy()
        #print(self.cell.final_entropy)
        # penal_stddev = np.std(self.cell.prob_attrac[1:])
        penal_stddev = self.cell.std_around_value()

        entropy = (-1.*self.cell.final_entropy ) - (self.cell.unclustered_cells * 0.01) # - (self.cell.penal * 0.01) #- (penal_stddev * 0.01)
        #if (self.cell.n_attrac <= 2.):
        #    max_entropy = 1
        #else:
        #    max_entropy = np.log2(self.cell.n_attrac)
        return entropy    #/max_entropy
    