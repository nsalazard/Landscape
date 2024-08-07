import matplotlib.pyplot as plt
import numpy as np
import glob
import os

state_colors = ['#000000', # Black
    '#DC143C', # Crimson
    '#00008B', # Dark Blue
    '#228B22', # Forest Green
    'indianred',
    'tab:blue',
    'tab:green',
    'gold',
    'tab:purple',
    '#FF4500', # Orange Red
    '#8A2BE2', # Blue Violet
    '#5F9EA0', # Cadet Blue
    '#D2691E', # Chocolate
    '#FF7F50', # Coral
    '#6495ED', # Cornflower Blue
    '#9932CC', # Dark Orchid
    '#E9967A', # Dark Salmon
    '#8FBC8F', # Dark Sea Green
    '#483D8B', # Dark Slate Blue
    ]

def d1_distance(result, target, weights):
    weights = np.tile(weights, (result.shape[0], 1))
    return np.sum(weights * np.abs(np.abs(result - target))) / 2.


def smooth_distribution(prob, eps=0.001):
    bins = prob.shape[1]
    empty_bins = np.sum(prob == 0., axis=1)
    smoothing = np.zeros(prob.shape)
    smoothing = (smoothing.T - empty_bins * eps / (bins - empty_bins)).T
    smoothing[prob == 0.] = eps
    return prob + smoothing


def kl_distance(result, target, weights):
    result_smoothed = smooth_distribution(result)
    # target_smoothed = smooth_distribution(target)
    kl = np.sum(np.where(target != 0., target * np.log(target / result_smoothed), 0.))
    return kl

#new
def plot_cell_concentration(data, state_names = None, row_labels=None, prob=False, output_dir =None):
    if row_labels is None:
        row_labels = [str(i) for i in range(1, data.shape[0] + 1)]
    state_names = [str(i) for i in range(1, data.shape[1] + 1)]
    data_cumulative = data.cumsum(axis=1)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(state_names, state_colors)):
        widths = data[:, i]
        starts = data_cumulative[:, i] - widths
        rects = ax.barh(row_labels, widths, left=starts, height=0.7, label=colname,
                        color=color, alpha=0.35)

        # if prob:
        #     bar_labels = [str(int(val*100)) for val in rects.datavalues]
        # else:
        bar_labels = [str(int(val * 100.)) for val in rects.datavalues]
        for b in range(len(bar_labels)):
            if bar_labels[b] == '0' or bar_labels[b] == '0':
                bar_labels[b] = ''
        ax.bar_label(rects, bar_labels, label_type='center', color='k')
        ax.legend(ncol=len(state_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    if output_dir is not None:
        output_path = os.path.join(output_dir, f"Proportions.png")
        plt.savefig(output_path)
    plt.show()


def plot_cell_proportions(data, state_names, state_colors, row_labels=None, prob=False):
    if row_labels is None:
        if data.shape[0] == 7:
            row_labels = ['D2', 'D2.5', 'D3', 'D3.5', 'D4.', 'D4.5', 'D5']  ###
        elif data.shape[0] == 8:
            row_labels = ['D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4.', 'D4.5', 'D5']  ###
        elif data.shape[0] == 9:
            row_labels = ['D1', 'D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4.', 'D4.5', 'D5']  ###
    data_cumulative = data.cumsum(axis=1)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    scale = np.sum(data[0])

    for i, (colname, color) in enumerate(zip(state_names, state_colors)):
        widths = data[:, i]
        starts = data_cumulative[:, i] - widths
        rects = ax.barh(row_labels, widths, left=starts, height=0.7, label=colname,
                        color=color, alpha=0.35)

        # if prob:
        #     bar_labels = [str(int(val*100)) for val in rects.datavalues]
        # else:
        bar_labels = [str(int(val * 100. / scale)) for val in rects.datavalues]
        for b in range(len(bar_labels)):
            if bar_labels[b] == '0' or bar_labels[b] == '0':
                bar_labels[b] = ''
        ax.bar_label(rects, bar_labels, label_type='center', color='k')
        ax.legend(ncol=len(state_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    plt.show()


def get_cell_data(filename):
    cell_data = np.loadtxt(filename) * 100
    cell_data = np.delete(cell_data, 2, axis=1)
    # move CE to after Tr instead
    # col_labels = ['EPI', 'Tr', 'PN', 'M', 'CE']
    idx = [0, 1, 4, 2, 3]
    cell_data = cell_data[:, idx]
    # col_labels = ['EPI', 'Tr', 'CE', 'PN', 'M']
    # print(col_labels)
    # print(cell_data)
    # col_colors = ['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue']
    return cell_data


def integrate_EM(f, t0, tf, nt, init_cond, noise, ndt=10, args=()):
    y = np.reshape(init_cond, (2, -1, 1))  # temporary solution ?
    # y = np.array(init_cond)
    # res = np.empty((len(init_cond), nt), dtype='float')
    res = np.empty((*y.shape, nt), dtype='float')

    t = t0

    res[:, 0] = y

    Delta_t = (tf - t0) / (nt - 1)
    dt = Delta_t / ndt
    sqrt_dt = np.sqrt(dt)

    for Delta_step in range(1, nt):
        for dt_step in range(ndt):
            # print('upd', upd.shape)
            y = y + f(t, y, *args) * dt + noise * np.random.standard_normal(y.shape) * sqrt_dt
            # print('y updated', y.shape)
            t += dt
        res[:, :, :, Delta_step] = y  # temporary
    return res

def delete_all_images(output_dir, file_extension='*.png'):
    # Construct the path pattern for the files to delete
    pattern = os.path.join(output_dir, file_extension)
    
    # List all files matching the pattern
    files_to_delete = glob.glob(pattern)
    
    # Delete each file
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            #print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
