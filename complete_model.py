from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io
from sklearn.model_selection import train_test_split
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import argparse
import numpy as np
import deepxde as dde 
# dde version 0.11
from deepxde.backend import tf
# from plotting_outcome import plot_losshistory, plot_beststate, plot_observed_predict

# parse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', dest='file_name', required = True, type = str, help='File name for input data')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-d', '--dimension', dest='dim', required = True, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    args = parser.parse_args()

# Network Parameters
num_hidden_layer_1d = 3 # number of hidden layers for NN (1D)
hidden_layer_size_1d = 20 # size of each hidden layers (1D)
num_hidden_layer_2d = 4 # number of hidden layers for NN (2D)
hidden_layer_size_2d = 36 # size of each hidden layers (2D)
num_domain = 10000 # number of training points within the domain
num_boundary = 1000 # number of training boundary condition points on the geometry boundary
num_initial = 0 # number of training initial condition points
num_test = 1000 # number of testing points within the domain
epochs = 40000 # number of epochs for training
lr = 0.001 # learning rate
noise = 0.1 # noise factor
loss_limit = 10 # upper limit to the initialized loss
test_size = 0.2 # precentage of testing data

# PDE Parameters
a = 0.01
b = 0.15
D = 0.1
k = 8
mu_1 = 0.2
mu_2 = 0.3
epsilon = 0.002

# Geometry Parameters
min_x = 0.1
max_x = 10            
min_y = 0.1
max_y = 10
min_t = 1
max_t = 70

def params_to_inverse(a,b,D,param):
    params = []
    if not param:
        return a, b, D, params
    # If inverse: 
    # The tf.variables are initialized with a positive scalar, relatively close to their ground truth values
    if 'a' in param:
        a = tf.math.exp(tf.Variable(-3.92))
        params.append(a)
    if 'b' in param:
        b = tf.math.exp(tf.Variable(-1.2))
        params.append(b)
    if 'd' in param:
        D = tf.math.exp(tf.Variable(-1.6))
        params.append(D)
    return a ,b , D, params

a, b, D, params = params_to_inverse(a,b,D,args.inverse)

def gen_data(file_name, dim, add_noise):
    
    data = scipy.io.loadmat(file_name)
    if dim == 1:
        t, x, Vsav, Wsav = data["t"], data["x"], data["Vsav"], data["Wsav"]
        X, T = np.meshgrid(x, t)
    elif dim == 2:
       t, x, y, Vsav, Wsav = data["t"], data["x"], data["y"], data["Vsav"], data["Wsav"]
       X, T, Y = np.meshgrid(x,t,y)
       Y = np.reshape(Y, (-1, 1))
    else:
        raise ValueError('Dimesion value argument has to be either 1 or 2')
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    V = np.reshape(Vsav, (-1, 1))
    W = np.reshape(Wsav, (-1, 1))    
    # With noise
    if add_noise:
        V = V + noise*np.random.randn(V.shape[0], V.shape[1])
    if dim == 1:     
        return np.hstack((X, T)), V, W
    return np.hstack((X, Y, T)), V, W

def pde_1D(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
    # Coupled PDE+ODE Equations
    eq_a = dv_dt -  D*dv_dxx + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

def pde_2D(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
    # Coupled PDE+ODE Equations
    eq_a = dv_dt -  D*(dv_dxx + dv_dyy) + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

def pde_1D_2_cycle(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
    
    x_space,t = x[:, 0:1],x[:, 1:2]
    t_stim_1 = tf.equal(t, 0)
    t_stim_2 = tf.equal(t, max_t)
    
    x_stim = tf.less_equal(x_space, 5*0.1)
    first_cond_stim = tf.logical_and(t_stim_1, x_stim)
    second_cond_stim = tf.logical_and(t_stim_2, x_stim)
    
    I_stim = tf.ones_like(x_space)*0.1
    I_not_stim = tf.ones_like(x_space)*0
    Istim = tf.where(tf.logical_or(first_cond_stim,second_cond_stim),I_stim,I_not_stim)
    # Coupled PDE+ODE Equations
    eq_a = dv_dt -  D*dv_dxx + k*V*(V-a)*(V-1) +W*V -Istim
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

def boundary_func_2d(x, on_boundary):
        return on_boundary and ~(x[0:2] == [min_x,min_y]).all() and  ~(x[0:2] == [min_x,max_y]).all() and ~(x[0:2] == [max_x,min_y]).all()  and  ~(x[0:2] == [max_x,max_y]).all() 
   
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
        raise ValueError('Dimesion value argument has to be either 1 or 2')
    return geomtime


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
    observe_v = dde.PointSetBC(observe_train, V_train, component=0)
    input_data = [bc_a, ic1, observe_v]
    # If W required as input
    if args.w_input:
        observe_w = dde.PointSetBC(observe_train, W_train, component=1)
        input_data = [bc_a, ic1, observe_v, observe_w]
    
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
                            num_test=num_test)    
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    
    # Stabalize initialization process
    losshistory, _ = model.train(epochs=1)
    num_itr = len(losshistory.loss_train)
    init_loss = max(losshistory.loss_train[num_itr-1])
    while init_loss>loss_limit or np.isnan(init_loss):
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        losshistory, _ = model.train(epochs=1)
        num_itr = len(losshistory.loss_train)
        init_loss = max(losshistory.loss_train[num_itr-1])
    
    # Train Network
    out_path = dir_path + args.model_folder_name
    if not args.inverse:
        losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path)
    else:
        variables_file = "variables_" + args.inverse + ".dat"
        variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)    
        losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    # Plot
    model_pred = model.predict(observe_test)
    v_pred = model_pred[:,0:1]
    rmse_v = np.sqrt(np.square(v_pred - V_test).mean())
    print("V rMSE test:", rmse_v)
    return train_state, v_pred, V_test 

# Run main code
train_state, V_pred, V_test = main(args)
