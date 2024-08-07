import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random
import os

from cell_class import Cells
from class_population import Population
from land_dataset_fitness import CellDiff_Dataset_Landscape
from morphogen_regimes import *
from landscape_visuals import *
from helper_func import plot_cell_proportions, get_cell_data
from landscape_segmentation import Somitogenesis_Landscape  #########
from class_module import Node, UnstableNode, Center, NegCenter

save_dir = 'saved_files/'
dir_img = 'images/1/'


if __name__ == '__main__':

        if not os.path.exists(dir_img):
            # Create the folder
            os.makedirs(dir_img)
            print(f"Folder '{dir_img}' created.")

        # MARK: - Cell Initialization

        t0 = 0.
        tf = 60.
        tc = 52.
        div = 40 #40 cells
        repl = 20 #20 replicates
        #Total num cell = 40*20 = 800
        nt = int(tf*3)
        noise_init = 0.5
        init_cond=(-8, 0)

        cell = Cells(t0 = t0, tf = tf, tc = tc , div = div,repl = repl, nt = nt, init_cond = init_cond)
        cell.create_Start_Times()
        cell.init_position(noise=noise_init)

        time_pars = (t0, tf, nt)
        morphogen_times = (tc,)

        # MARK: - Parameters

        par_limits = {
            'x': (-10.,10.),
            'y': (-5., 5.),
            'a': (0.2,3.),
            's': (0.2, 2)
        }

        par_choice_values = {
            'tau': (5.,),
            'tilt_lmt': (-0.41, -0.01),
        }

        landscape_pars = {
            'A0': 0.00005,
            'init_cond': (0., 0.),
            'regime': mr_sigmoid,
            'n_regimes': 2,
            'morphogen_times': morphogen_times,
            'used_fp_types': (Node,),
            'immutable_pars_list': [],
            'tilt': -0.35,
            'tilt_par': (0.5)
        }

        prob_pars = {
            'prob_tilt': 0.10,
            'prob_add': 0.15,
            'prob_drop': 0.15,
            'prob_shuffle': 0.
            # the rest is mutation of parameters
        }

        fitness_pars = {
            'ncells': 50,
            'time_pars': time_pars,
            'init_state': (0., 0.),
            't0_shift': 0.5,  # shift (delay) of the time of transition between 2 neighbor cells
            'noise': 0.3,
            'low_value': -1.,
            'high_value': 1.,
            'penalty_weight': 0.1,
            't_stable': 5, # how many timepoints should be at steady state
            'ndt': 50,
            'tilt': (-0.001, -0.4)
        }

        # MARK: - Start_module_list


        #  Starting with 2 random nodes, then any modules can be added or deleted
        start_module_list = [Node.generate(par_limits, par_choice_values, n_regimes=2) for i in range(2)]

        # Population sizeshould be even, adjust N to your computing capacity
        N = 8
        P = Population(cell = cell,N = N, problem_type = Somitogenesis_Landscape, landscape_pars = landscape_pars, 
                    prob_pars = prob_pars, fitness_pars = fitness_pars, par_limits = par_limits, 
                    par_choice_values = par_choice_values, start_module_list = start_module_list)


        fitness_traj = P.evolve_parallel(3, fitness_pars, save_dir, save_each=5)
        print('Done')

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        plt.figure(figsize=(4,3))
        plt.plot(fitness_traj, lw=2, c='steelblue')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best fitness', fontsize=12)
        filename = f"{timestamp}-Generation_BFitness.png"
        file_path = os.path.join(dir_img, filename)
        plt.savefig(file_path)
        #plt.show()
        plt.close()

        plt.imshow(P.landscape_list[0].result, cmap='Blues')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Space', fontsize=12)
        filename = f"{timestamp}-Time_Space.png"
        file_path = os.path.join(dir_img, filename)
        plt.savefig(file_path)
        #plt.show()
        plt.close()

        plt.plot(P.landscape_list[0].result[:, -1], lw=2)
        plt.title('Final pattern', fontsize=12)
        plt.xlabel('Space', fontsize=12)
        filename = f"{timestamp}-FinalPattern_Space.png"
        file_path = os.path.join(dir_img, filename)
        plt.savefig(file_path)
        #plt.show()
        plt.close()


        plt.plot(P.landscape_list[0].result[20, :], lw=2)
        plt.title('Single-cell dynamics', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        filename = f"{timestamp}-Cell_Time.png"
        file_path = os.path.join(dir_img, filename)
        plt.savefig(file_path)
        #plt.show()
        plt.close()
