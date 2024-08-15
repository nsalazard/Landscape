import multiprocessing as mp
import os
import pickle
import time
from copy import deepcopy
import numpy as np
from cell_class import Cells
from landscape_visuals import visualize_potential

class Population:
    def __init__(self, cell: Cells, N: int, problem_type, landscape_pars, prob_pars, fitness_pars,
                 par_limits, par_choice_values, start_module_list=(), start_fitness=-np.inf):
        
        self.cell = cell
        self.N = N  # N >= 1 !
        self.problem_type = problem_type
        self.landscape_pars = landscape_pars
        self.prob_pars = prob_pars
        self.par_limits = par_limits
        self.par_choice_values = par_choice_values

        self.landscape_list = []
        for i in range(N):
            self.landscape_list.append(self.problem_type(cell,
                                                         start_module_list, landscape_pars['A0'],
                                                         landscape_pars['init_cond'],
                                                         landscape_pars['regime'],
                                                         landscape_pars['n_regimes'],
                                                         landscape_pars['morphogen_times'],
                                                         landscape_pars['used_fp_types'],
                                                         landscape_pars['immutable_pars_list'],
                                                         landscape_pars['tilt'],
                                                         landscape_pars['tilt_par'])
                                                        )

        if start_module_list and fitness_pars:
            fitness = self.landscape_list[0].get_fitness(fitness_pars)
        else:
            fitness = start_fitness
        for landscape in self.landscape_list:
            landscape.fitness = fitness
            print(fitness, end  = ' ')

# ____________________________________________________________________________________________________________________
#  MARK: - Evolve

    def evolve(self, ngenerations, fitness_pars, saved_files_dir, save_each=10):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print('Timecode:', timestr)

        os.makedirs(saved_files_dir + self.problem_type.__name__ + '/' + timestr)
        save_dir = saved_files_dir + self.problem_type.__name__ + '/' + timestr + '/'
        save_gens_file = open(save_dir + timestr + "_generations.txt", "a")
        pickle_name = save_dir + timestr + "_module_list_"

        save_gens_file.write('# Evolution of ' + str(self.N) + ' landscapes for ' + self.problem_type.__name__ + '\n'
                             + '# Starting: ' + str(
            max([landscape.fitness for landscape in self.landscape_list])) + '\n')
        with open(save_dir + timestr + "_parameters.pickle", "wb") as f:
            pickle.dump([self.landscape_pars, self.prob_pars,
                         fitness_pars, self.par_limits, self.par_choice_values], f)
        # with open(pickle_name + 'initial' + '.pickle', "wb") as f:
        #     pickle.dump(self.landscape_list[0].module_list, f)
        with open(save_dir + timestr + "_initial_full.pickle", "wb") as f:
            pickle.dump(self, f)

        for igen in range(ngenerations):
            for odd_landscape in self.landscape_list[::2]:
                odd_landscape.mutate(self.par_limits, self.par_choice_values, self.prob_pars, fitness_pars)
            self.landscape_list.sort(key=lambda landscape: landscape.fitness + 0.002 * np.random.randn(), reverse=True)
            del (self.landscape_list[self.N // 2:])
            self.landscape_list = [deepcopy(landscape) for landscape in self.landscape_list for _ in range(2)]

            save_gens_file.write(str(igen) + ' ' + str(np.round(self.landscape_list[0].fitness, 4)) + ' '
                                 + str(len(self.landscape_list[0].module_list)) + '\n')
            if igen % save_each == 0:
                with open(pickle_name + str(igen) + '.pickle', "wb") as f:
                    pickle.dump(self.landscape_list[0].module_list, f)
        save_gens_file.close()
        with open(save_dir + timestr + "_result_full.pickle", "wb") as f:
            pickle.dump(self, f)
        print('Best fitness:', max([landscape.fitness for landscape in self.landscape_list]))

# ______________________________________________________________________________________________________________________
#  MARK: - Evolve Parallel

    def evolve_parallel(self, ngenerations, fitness_pars, saved_files_dir, save_each=10,img_save=20, output_dir=None):
        """ Evolutionary optimization using all CPUs """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print('Timecode:', timestr)
        fitness_traj = np.zeros(ngenerations)
        os.makedirs(saved_files_dir + self.problem_type.__name__ + '/' + timestr)
        save_dir = saved_files_dir + self.problem_type.__name__ + '/' + timestr + '/'
        save_gens_file = open(save_dir + timestr + "_generations.txt", "a")
        pickle_name = save_dir + timestr + "_module_list_"
        save_gens_file.write(
            '# Parallel evolution of ' + str(self.N) + ' landscapes for ' + self.problem_type.__name__ + '\n'
            + '#Starting: ' + str(max([landscape.fitness for landscape in self.landscape_list])) + '\n')
        with open(save_dir + timestr + "_parameters.pickle", "wb") as f:
            pickle.dump([self.landscape_pars, self.prob_pars, fitness_pars,
                         self.par_limits, self.par_choice_values], f)
        with open(save_dir + timestr + "_initial_full.pickle", "wb") as f:
            pickle.dump(self, f)

        pool = mp.Pool(mp.cpu_count())

        for igen in range(ngenerations):
            if igen % 10 == 0:
                print('Generation:', igen)
            results = []
            for odd_landscape in self.landscape_list[::2]:
                results.append(pool.apply_async(odd_landscape.mutate_and_return,
                                                args=(self.par_limits, self.par_choice_values,
                                                      self.prob_pars, fitness_pars)))

            for ind in range(self.N // 2):
                result = results[ind].get()
                #print(f'ind = {ind}   result = {result}')
                self.landscape_list[2 * ind] = result

            self.landscape_list.sort(key=lambda landscape: landscape.fitness + 0.002 * np.random.randn(), reverse=True)

            #             print('after mutation:', [landscape.fitness for landscape in self.landscape_list])
            del (self.landscape_list[self.N // 2:])
            fitness_traj[igen] = self.landscape_list[0].fitness
            self.landscape_list = [deepcopy(landscape) for landscape in self.landscape_list for _ in range(2)]
            save_gens_file.write(str(igen) + ' ' + str(np.round(self.landscape_list[0].fitness, 4)) + ' '
                                 + str(len(self.landscape_list[0].module_list)) + '\n')

            if igen % save_each == 0:
                with open(pickle_name + str(igen) + '.pickle', "wb") as f:
                    pickle.dump(self.landscape_list[0].module_list, f)
            
            if output_dir != None and (igen % img_save == 0 or igen == ngenerations - 1):
                L = 10.
                npoints = 201
                q = np.linspace(-L, L, npoints)
                xx, yy = np.meshgrid(q, q, indexing='xy')
                fig = visualize_potential(self.landscape_list[0], xx, yy, regime= self.landscape_list[0].cell.tf, 
                                          color_scheme='order', scatter=True, elev=20, azim=-90, output_gif = output_dir, 
                                          igen =igen, fit = self.landscape_list[0].fitness)

            

        save_gens_file.close()
        pool.close()

        pool.join()

        with open(save_dir + timestr + "_result_full.pickle", "wb") as f:
            pickle.dump(self, f)
        print('Best fitness:', max([landscape.fitness for landscape in self.landscape_list]))
        return fitness_traj


