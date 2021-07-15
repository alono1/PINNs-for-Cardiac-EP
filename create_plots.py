import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

def plot_1d(data_list, model, fig_name):

    plot_1d_cell(data_list, model, fig_name[1:])
    plot_1d_array(data_list, model, fig_name[1:])
    return 0
    
def plot_1d_cell(data_list, model, fig_name):
    
    ## Unpack data
    observe_x, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]
    
    ## Pick a random cell to show
    obs_x = 15.0
    
    ## Get data for cell
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][0]==obs_x]
    observe_geomtime = observe_x[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,1]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if observe_train[i][0]==obs_x]
    v_trained_points = v_train[idx_train]
    t_markers = (observe_train[idx_train])[:,1]
    
    ## create figure
    plt.figure()
    ## If there are any trained data points for the current cell 
    if len(t_markers):
        plt.scatter(t_markers, v_trained_points, marker='x', c='black',s=4, label='Observed')
    plt.plot(t_axis, v_GT, c='b', label='GT')
    plt.plot(t_axis, v_predict, c='r', label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_cell_plot.tiff")
    png1.close()
    return 0

def plot_1d_array(data_list, model, fig_name):
    
    ## Unpack data
    observe_x, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]
    
    ## Pick a random point in time to show
    obs_t = 36.0
    
    ## Get all array data for chosen time 
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][1]==obs_t]
    observe_geomtime = observe_x[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    x_ax = observe_geomtime[:,0]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if observe_train[i][1]==obs_t]
    v_trained_points = v_train[idx_train]
    x_markers = (observe_train[idx_train])[:,0]

    ## create figure
    plt.figure()
    ## If there are any trained data points for the current time step
    if len(x_markers):
        plt.scatter(x_markers, v_trained_points, marker='x', c='black',s=4, label='Observed')
    plt.plot(x_ax, v_GT, c='b', label='GT')
    plt.plot(x_ax, v_predict, c='r', label='Predicted')
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_array_plot.tiff")
    png1.close()
    return 0
