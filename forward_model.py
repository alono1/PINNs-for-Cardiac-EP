from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io

import argparse

# Adding the path. Only needed for windows
import sys
sys.path.append('D:\\Alon\\My Studies\\M.SC Artificial Intelligence\\Individual Project')

import numpy as np
import deepxde as dde
# from deepxde.backend import tf
from plotting_outcome import plot_losshistory, plot_beststate, plot_observed_predict

def gen_traindata(file_name, add_noise = False, noise = 0.0, v_std = 0.1):
    
    data = scipy.io.loadmat(file_name)
    t, x, Vsav, Wsav = data["t"], data["x"], data["Vsav"], data["Wsav"]
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    V = np.reshape(Vsav, (-1, 1))
    W = np.reshape(Wsav, (-1, 1))    
    # With noise
    if add_noise:
        # revisit the std value
        V = V + noise*v_std*np.random.randn(V.shape[0], V.shape[1])
    return np.hstack((X, T)), V, W

def pde(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
    a = 0.01
    b = 0.15
    k = 8
    mu_1 = 0.2
    mu_2 = 0.3
    D = 0.1
    epsilon = 0.002
    eq_a = dv_dt -  D*dv_dxx + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

def geometry_time(dim, observe_x):
    if dim == 1:
        geom = dde.geometry.Interval(0.1, 20)
        timedomain = dde.geometry.TimeDomain(1, 70)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)    
    elif dim ==2:
        min_x = np.min(observe_x[:,0:1])
        max_x = np.max(observe_x[:,0:1])           
        min_y = np.min(observe_x[:,1:2])
        max_y = np.max(observe_x[:,1:2])
        min_t = np.min(observe_x[:,2:3])
        max_t = np.max(observe_x[:,2:3])
        geom = dde.geometry.Rectangle([min_x,min_y], [max_x,max_y])
        timedomain = dde.geometry.TimeDomain(min_t, max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    else:
        raise ValueError('The entered dimesion value has to be either 1 or 2')
    return (geom, timedomain, geomtime)

def main(args):
    # Generate Data 
    add_noise = False
    if args.noise:
        add_noise = True
    noise = args.noise
    file_name = args.file_name
    observe_x, V, W = gen_traindata(file_name, add_noise, noise)
        
    # Geometry and Time domains
    geom, timedomain, geomtime = geometry_time(args.dim, observe_x)
    
    # Change boundary conditions
    bc_a = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        
    # Model observed data
    observe_y1 = dde.PointSetBC(observe_x, V, component=0)
    input_data = [bc_a,observe_y1]
    # If include W in input
    if args.w_input:
        observe_y2 = dde.PointSetBC(observe_x, W, component=1)
        input_data = [bc_a, observe_y1, observe_y2]
        
    data = dde.data.TimePDE(geomtime, pde, input_data, num_domain=10000, num_boundary=1400, anchors=observe_x,
        train_distribution="uniform", num_test=4000)
    # Define Network
    if args.dim ==1:
        net = dde.maps.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")
    elif args.dim ==2:
        net = dde.maps.FNN([3] + [20] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    
    # Train Network
    losshistory, train_state = model.train(epochs=80000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    # Plots
    y_pred = model.predict(observe_x)
    f = model.predict(observe_x, operator=pde)
    print("Mean residual:", np.mean(np.absolute(f)))
    y_true=np.concatenate((V, W), axis=1)
    print("L2 relative error:", dde.metrics.l2_relative_error( y_true, y_pred))
    np.savetxt("true_pred.dat", np.hstack((observe_x, y_true, y_pred)),header="observe_x,y_true, y_pred")
    plot_losshistory(losshistory)
    # plot_beststate(train_state)
    # plot_observed_predict(observe_x,y_true, y_pred)
    return train_state, y_true, y_pred, f

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--file-name', dest='file_name', required = True, type = str, help='File name for input data')
    parser.add_argument('-d', '--dimension', dest='dim', required = True, type = int, default = 2, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', required = False, default = 0.0, type = float, help='Add noise to the data')
    parser.add_argument('-w', '--w-input',   dest='w_input',   action='store_true', help='Add W to the model input data')
    args = parser.parse_args()

    train_state, y_true, y_pred, f=main(args)

