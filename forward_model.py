from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io
from sklearn.model_selection import train_test_split
# File directory
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import argparse

# Adding the path. Only needed for windows
import sys
sys.path.append(dir_path)

import numpy as np
import deepxde as dde 
# dde version 0.11
# from deepxde.backend import tf
from plotting_outcome import plot_losshistory, plot_beststate, plot_observed_predict

# Network Parameters
num_hidden_layer_1d = 3 # number of hidden layers for NN (1D)
hidden_layer_size_1d = 20 # size of each hidden layers (1D)
num_hidden_layer_2d = 4 # number of hidden layers for NN (2D)
hidden_layer_size_2d = 36 # size of each hidden layers (2D)
num_domain = 0 # number of training points within the domain
num_boundary = 0 # number of training boundary condition points on the geometry boundary
num_initial = 0 # number of training initial condition points
num_test = 4000 # number of testing points within the domain
epochs = 40000 # number of epochs for training
lr = 0.001 # learning rate
noise = 0.1 # noise factor
test_size = 0.2 # precentage of testing data out of the whole data file

# PDE Parameters
a = 0.01
b = 0.15
k = 8
mu_1 = 0.2
mu_2 = 0.3
D = 0.1
epsilon = 0.002

# Geometry Parameters
min_x = 0.1
max_x = 10            
min_y = 0.1
max_y = 10
min_t = 1
max_t = 70

def gen_data(file_name, dim, add_noise, v_std = 0.1):
    
    data = scipy.io.loadmat(file_name)
    if dim == 1:
        t, x, Vsav, Wsav = data["t"], data["x"], data["Vsav"], data["Wsav"]
        X, T = np.meshgrid(x, t)
    elif dim == 2:
       t, x, y, Vsav, Wsav = data["t"], data["x"], data["y"],data["Vsav"], data["Wsav"]
       X, T, Y = np.meshgrid(x,t,y)
       Y = np.reshape(Y, (-1, 1))
    else:
        raise ValueError('The entered dimesion value has to be either 1 or 2')
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    V = np.reshape(Vsav, (-1, 1))
    W = np.reshape(Wsav, (-1, 1))    
    # With noise
    if add_noise:
        V = V + noise*v_std*np.random.randn(V.shape[0], V.shape[1])
    if dim == 1:     
        return np.hstack((X, T)), V, W
    return np.hstack((X, Y, T)), V, W

def pde_1D(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
    eq_a = dv_dt -  D*dv_dxx + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

def pde_2D(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
    eq_a = dv_dt -  D*(dv_dxx + dv_dyy) + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

def geometry_time(dim, observe_x):
    if dim == 1:
        geom = dde.geometry.Interval(min_x, 2*max_x)
        timedomain = dde.geometry.TimeDomain(min_t, max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)    
    elif dim == 2:
        geom = dde.geometry.Rectangle([min_x,min_y], [max_x,max_y])
        timedomain = dde.geometry.TimeDomain(min_t, max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    else:
        raise ValueError('The entered dimesion value has to be either 1 or 2')
    return geomtime

def boundary_func_2d(x, on_boundary):
        return on_boundary and ~(x[0:2] == [min_x,min_y]).all() and  ~(x[0:2] == [min_x,max_y]).all() and ~(x[0:2] == [max_x,min_y]).all()  and  ~(x[0:2] == [max_x,max_y]).all() 


def main(args):

    # Generate Data 
    add_noise = args.noise
    file_name = args.file_name
    observe_x, V, W = gen_data(file_name, args.dim, add_noise)
    # Split data to train and test
    observe_train, observe_test, V_train, V_test, W_train, W_test = train_test_split(observe_x,V,W,test_size=test_size)
    
    # Define Initial Conditions
    T_ic = observe_train[:,-1].reshape(-1,1)
    idx_init = np.where(np.isclose(T_ic,1))[0]
    V_init = V_train[idx_init]
    observe_init = observe_train[idx_init]
    ic1 = dde.PointSetBC(observe_init,V_init,component=0)
    
    # Geometry and Time domains
    geomtime = geometry_time(args.dim, observe_x)
    
    # Define Boundary Conditions
    if args.dim == 1:
        bc_a = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
    elif args.dim == 2:
        bc_a = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), boundary_func_2d, component=0)
    
    # Model observed data
    observe_y1 = dde.PointSetBC(observe_train, V_train, component=0)
    input_data = [bc_a, ic1, observe_y1]
    # If include W in input
    if args.w_input:
        observe_y2 = dde.PointSetBC(observe_train, W_train, component=1)
        input_data = [bc_a, ic1, observe_y1, observe_y2]
    
    if args.dim == 1:
        # Select relevant PDE
        pde = pde_1D
        # Define the Network
        net = dde.maps.FNN([2] + [hidden_layer_size_1d] * num_hidden_layer_1d + [2], "tanh", "Glorot uniform")
    elif args.dim == 2:
        # Select relevant PDE
        pde = pde_2D
        # Define the Network
        net = dde.maps.FNN([3] + [hidden_layer_size_2d] * num_hidden_layer_2d + [2], "tanh", "Glorot uniform") 
    data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain, 
                            num_boundary=num_boundary, 
                            anchors=observe_train,
                            # train_distribution="uniform",
                            num_test=num_test)    
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
        
    # Train Network
    losshistory, train_state = model.train(epochs=epochs, model_save_path = dir_path + args.model_folder_name)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    # Plots
    model_pred = model.predict(observe_test)
    V_pred = model_pred[:,0]
    v_rmse = np.sqrt(np.square(V_pred - V_test).mean())
    print("V rMSE :", v_rmse)
    #f = model.predict(observe_x, operator = pde)
    #print("Mean residual:", np.mean(np.absolute(f)))
    #y_true=np.concatenate((V, W), axis=1)
    #print("L2 relative error:", dde.metrics.l2_relative_error( y_true, y_pred))
    # plot_losshistory(losshistory)
    # np.savetxt("true_pred.dat", np.hstack((observe_x, y_true, y_pred)),header="observe_x,y_true, y_pred")
    # plot_beststate(train_state)
    # plot_observed_predict(observe_x,y_true, y_pred)
    return train_state, V_pred, V_test 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', dest='file_name', required = True, type = str, help='File name for input data')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-d', '--dimension', dest='dim', required = True, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    args = parser.parse_args()
    
    train_state, V_pred, V_test = main(args)
