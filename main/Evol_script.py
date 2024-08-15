import matplotlib.pyplot as plt
from datetime import datetime
#from IPython.display import HTML
import numpy as np
import random
import os

from cell_class import Cells
from class_population import Population
from land_dataset_fitness import CellDiff_Dataset_Landscape
from morphogen_regimes import *
from landscape_visuals import *
from helper_func import plot_cell_concentration, get_cell_data, delete_all_images, create_directory_if_not_exists
from landscape_segmentation import Somitogenesis_Landscape  #########
from class_module import Node, UnstableNode, Center, NegCenter

seed= 22
NUM_EVO = 10
NUM_LAND = 16
save_dir = f'saved_files_{seed}/'

np.random.seed(seed)
from decimal import Decimal, getcontext, ROUND_HALF_UP
getcontext().prec = 6
getcontext().rounding = ROUND_HALF_UP


if __name__ == '__main__':
    from cell_class import Cells
    t0 = 0.
    tf = 80.
    tc = 60.
    div = 40 #40 cells
    repl = 20 #20 replicates
    #Total num cell = 40*20 = 800
    nt = int(tf*3)
    noise_init = 0.5
    init_cond=(-8, 0)
    W_H_d = 0.5
    W_H_dp = 1.5

    cell = Cells(t0 = t0, tf = tf, tc = tc , div = div,repl = repl, nt = nt, init_cond = init_cond, W_H_d = 0.5, W_H_dp = 1.5)
    cell.create_Start_Times()
    cell.init_position(noise=noise_init)

    # Parameters

    time_pars = (t0, tf, nt)
    morphogen_times = (tc,)

    par_limits = {
        'x': (-8.,8.),
        'y': (-5., 5.),
        'a': (0.3,3.),
        's': (0.3, 2)
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

    # Initialization of the population

    #  Starting with 2 random nodes, then any modules can be added or deleted
    start_module_list = [Node.generate(par_limits, par_choice_values, n_regimes=2) for i in range(3)]

    # Population size should be even, adjust N to your computing capacity
    P = Population(cell = cell,N = NUM_LAND, problem_type = Somitogenesis_Landscape, landscape_pars = landscape_pars, prob_pars = prob_pars, fitness_pars = fitness_pars, par_limits = par_limits, par_choice_values = par_choice_values, start_module_list = start_module_list)

    #Creation of directories
    create_directory_if_not_exists(f"images/{seed}/")
    create_directory_if_not_exists(f'images/{seed}/evo/')
    create_directory_if_not_exists(f'images/{seed}/Gif/')
    create_directory_if_not_exists(f'images/{seed}/Cells/')

    #EVOLUTION

    fitness_traj = P.evolve_parallel(NUM_EVO, fitness_pars, save_dir, save_each=5, output_dir = f'images/{seed}/evo/')
    print('Done')

    output_dir = f"images/{seed}/"

    plt.figure(figsize=(4,3))
    plt.plot(fitness_traj, lw=2, c='steelblue')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best fitness', fontsize=12)
    # plt.ylim((-2,0))
    output_path = os.path.join(output_dir, f"Generation_BF.png")
    plt.savefig(output_path)
    #plt.show()

    plt.imshow(P.landscape_list[0].result, cmap='Blues')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Space', fontsize=12)
    output_path = os.path.join(output_dir, f"Time_Space.png")
    plt.savefig(output_path)
    #plt.show()

    plt.plot(P.landscape_list[0].result[:, -1], lw=2)
    plt.title('Final pattern', fontsize=12)
    plt.xlabel('Space', fontsize=12)
    output_path = os.path.join(output_dir, f"FinalPattern.png")
    plt.savefig(output_path)
    #plt.show()

    plt.plot(P.landscape_list[0].result[20, :], lw=2)
    plt.title('Single-cell dynamics', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    output_path = os.path.join(output_dir, f"SCdynamics.png")
    plt.savefig(output_path)
    #plt.show()

    P.landscape_list[0].cell.get_data_concentration()
    cell_data = P.landscape_list[0].cell.data_States
    plot_cell_concentration(cell_data, output_dir= output_dir)

    visualize_div_time(P.landscape_list[0], output_dir= output_dir)
    visualize_last_div_time(P.landscape_list[0], output_dir= output_dir)

    # Create a gif of the landscape through evolution

    create_gif_from_images(f'images/{seed}/evo/', f'images/{seed}/Land_evo.gif', duration=700)
    delete_all_images(f'images/{seed}/evo/', '*.png')

    # Create a gif of the landscape through time, in the last step of evolution

    L = 11.
    npoints = 201
    q = np.linspace(-L, L, npoints)
    xx, yy = np.meshgrid(q, q, indexing='xy')
    output_gif = f'images/{seed}/Gif/'
    for istep in range(0, int(P.landscape_list[0].cell.nt) + 1, 2):
        fig = visualize_potential(P.landscape_list[0], xx, yy, regime= istep, color_scheme='order', scatter=True,
                                elev=20, azim=-90, output_gif = output_gif, igen =150, fit = P.landscape_list[0].fitness)

    create_gif_from_images(output_gif, f'images/{seed}/potential_time.gif', duration=1000)
    delete_all_images(f'images/{seed}/Gif/', '*.png')

    #Create a video of the cell dynamics

    video_landscape(P.landscape_list[0], xx, yy, traj_times=None, plot_start=0, size_x=4, size_y=4, ndt= nt, noise_init=2., 
                noise_run=0.2, tstep = 0 ,color_scheme='fp_types', colors=None, video_name = 'Land_cells_n',
                  same_time=False, dwl= False, output_dir =f'images/{seed}/')

