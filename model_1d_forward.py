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


def gen_traindata(add_noise = False, noise = 0.0, v_std = 0.1):
    
    data = scipy.io.loadmat('trial_1D_RK.mat')
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
        # W=W+ noise*np.std(W)*np.random.randn(W.shape[0], W.shape[1])

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

def main(args):

    # Geometry and Time domains
    geom = dde.geometry.Interval(0.1, 20)
    timedomain = dde.geometry.TimeDomain(1, 70)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # Change boundary conditions
    bc_a = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        
    # Observed Data 
    if args.noise:
        add_noise = True
        noise = args.noise
    observe_x, V, W = gen_traindata(add_noise, noise)
    observe_y1 = dde.PointSetBC(observe_x, V, component=0)
    input_data = [bc_a,observe_y1]
    # If include W in input
    if args.w_input:
        observe_y2 = dde.PointSetBC(observe_x, W, component=1)
        input_data = [bc_a, observe_y1, observe_y2]
    
    data = dde.data.TimePDE(geomtime, pde, input_data, num_domain=10000, num_boundary=1400, anchors=observe_x,
        train_distribution="uniform", num_test=4000)

    # Define Network
    net = dde.maps.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    #variable = dde.callbacks.VariableValue([a], period=1000, filename="variables.dat")
    
    # Train Network
    # checkpointer = dde.callbacks.ModelCheckpoint("./model_end/model.ckpt", verbose=1, save_better_only=True)
    # callbacks=[checkpointer]
    # callbacks = [variable,checkpointer]
    # losshistory, train_state = model.train(epochs=80000, model_restore_path = "./model/model.ckpt-1", callbacks = callbacks)
    # losshistory, train_state = model.train(epochs=80000, callbacks = callbacks)
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

    parser.add_argument('-n', '--noise', dest='noise', required = False, default = 2, type = float, help='Add noise to the data')
    parser.add_argument('-w', '--w-input',   dest='w_input',   action='store_true', help='Add W to the input data')
    args = parser.parse_args()

    train_state, y_true, y_pred, f=main(args)
    X_train, y_train, X_test, y_test, best_y, best_ystd= train_state.packed_data()

