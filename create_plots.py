import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_1d(v, observe_x, model, fig_name):
    # t_min = observe_x[0][1]
    # t_max = observe_x[-1][1]
    # x_min = observe_x[0][0]
    # x_max = observe_x[-1][0]
    plot_1d_cell(v, observe_x, model, fig_name)
    # plot_1d_t(v_pred, v_test, observe_x, fig_name)
    return 0
    
def plot_1d_cell(v, observe_x, model, fig_name):
    
    # Pick a random cell to show
    obs_x = 6.0
    # Get data for cell
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][0]==obs_x]
    observe_test = observe_x[idx]
    v_test = v[idx]
    
    v_predict = model.predict(observe_test)[:,0:1]
    t_axis = observe_test[:,1]
    plt.figure()
    
    plt.plot(t_axis, v_test, c='r', label='observed')
    plt.plot(t_axis, v_predict, c='b', label='predicted')
    plt.legend(loc='upper right')
    # plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.show()
    plt.savefig(fig_name + "_cell_plot")
    return 0

def plot_1d_array(v, observe_x, model, fig_name):
    
    # Pick a random point in time to show
    obs_t = 6.0
    # Get data for chosen time 
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][1]==obs_t]
    observe_test = observe_x[idx]
    v_test = v[idx]
    v_predict = model.predict(observe_test)[:,0:1]
    x_ax = observe_test[:,0]
    # create and save plot
    plt.figure()
    
    plt.plot(x_ax, v_test, c='r', label='observed')
    plt.plot(x_ax, v_predict, c='b', label='predicted')
    plt.legend(loc='upper right')
    # plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('V')
    plt.show()
    plt.savefig(fig_name + "_array_plot")
    return 0


# def plot_check(v_predict, v_test, t_ax):
# #     plt.figure()
    
#     plt.plot(t_ax, v_test, c='r', label='observed')
#     plt.plot(t_ax, v_predict, c='b', label='predicted')
#     plt.legend(loc='best')
#     plt.xlabel('time')
#     plt.ylabel('V')
#     plt.savefig("checking")
#     plt.show()
#     return 0

# v_predict = np.random.randn(70)
# v_test = np.random.randn(70)
# t_ax = np.arange(0,70) 
# plot_check(v_predict,v_test,t_ax)

