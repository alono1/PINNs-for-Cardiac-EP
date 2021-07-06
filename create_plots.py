import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_1d(data_list, model, fig_name):

    plot_1d_cell(data_list, model, fig_name[1:-1])
    plot_1d_array(data_list, model, fig_name[1:-1])
    return 0
    
def plot_1d_cell(data_list, model, fig_name):
    
    # Unpack data
    observe_x, observe_train, v = data_list[0], data_list[1], data_list[2]
    
    # Pick a random cell to show
    obs_x = 15.0
    
    # Get data for cell
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][0]==obs_x]
    observe_geomtime = observe_x[idx]
    v_observe = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,1]
    
    # Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if observe_train[i][0]==obs_x]
    t_markers = ((observe_train[idx_train])[:,1]).astype(int) - 1
    
    # create and save plot
    plt.figure()
    plt.plot(t_axis, v_observe, c='b', label='observed', marker='x', markevery=t_markers, ms=5)
    plt.plot(t_axis, v_predict, c='r', label='predicted')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.title('Potential vs Time: Cell Plot')
    plt.savefig(fig_name + "_cell_plot.png")
    return 0

def plot_1d_array(data_list, model, fig_name):
    
    # Unpack data
    observe_x, observe_train, v = data_list[0], data_list[1], data_list[2]
    
    # Pick a random point in time to show
    obs_t = 36.0
    
    # Get all array data for chosen time 
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][1]==obs_t]
    observe_geomtime = observe_x[idx]
    v_observe = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    x_ax = observe_geomtime[:,0]
    
    # Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if observe_train[i][1]==obs_t]
    x_train = (((observe_train[idx_train])[:,0])*10).astype(int) -1

    # create and save plot
    plt.figure()
    plt.plot(x_ax, v_observe, c='b', label='observed',  marker='x', markevery=x_train, ms=5)
    plt.plot(x_ax, v_predict, c='r', label='predicted')
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('V')
    plt.title('Potential vs Space: Cable Plot')
    plt.savefig(fig_name + "_array_plot.png")
    return 0
