from sklearn.model_selection import train_test_split
import sys
import os         
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import argparse
import numpy as np
import deepxde as dde # version 0.11
from create_plots import plot_1d
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', dest='file_name', required = True, type = str, help='File name for input data')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-d', '--dimension', dest='dim', required = True, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    parser.add_argument('-p', '--plot', dest='plot', required = False, action='store_true', help='Create and save plots')
    args = parser.parse_args()

## Network Parameters
num_hidden_layers_1d = 4 # number of hidden layers for NN (1D)
hidden_layer_size_1d = 32 # size of each hidden layers (1D)
num_hidden_layers_2d = 4 # number of hidden layers for NN (2D)
hidden_layer_size_2d = 32 # size of each hidden layers (2D)
num_domain = 10000 # number of training points within the domain
num_boundary = 1000 # number of training boundary condition points on the geometry boundary
num_test = 1000 # number of testing points within the domain

## Training Parameters
MAX_MODEL_INIT = 10 # maximum number of times allowed to initialize the model
MAX_LOSS = 10 # upper limit to the initialized loss
epochs = 50000 # number of epochs for training
lr = 0.001 # learning rate
noise = 0.1 # noise factor
test_size = 0.9 # precentage of testing data


def main(args):
    
    ## Get utilities Class
    dynamics = utils.system_dynamics()
    params = dynamics.params_to_inverse(args.inverse)
    
    ## Generate Data 
    file_name = args.file_name
    observe_x, V, W = dynamics.generate_data(file_name, args.dim)  
    
    ## Split data to train and test
    observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=test_size)
    
    ## Add noise to training data if needed
    if args.noise:
        v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

    ## Define Initial Conditions
    T_ic = observe_train[:,-1].reshape(-1,1)
    idx_init = np.where(np.isclose(T_ic,1))[0]
    V_init = v_train[idx_init]
    observe_init = observe_train[idx_init]
    ic_1 = dde.PointSetBC(observe_init,V_init,component=0)
    
    ## Geometry and Time domains
    geomtime = dynamics.geometry_time(args.dim, observe_x)
    
    ## Define Boundary Conditions
    if args.dim == 1:
        bc_a = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
    elif args.dim == 2:
        bc_a = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), dynamics.boundary_func_2d, component=0)
    
    # Model observed data
    observe_v = dde.PointSetBC(observe_train, v_train, component=0)
    input_data = [bc_a, ic_1, observe_v]
    ## If W required as input
    if args.w_input:
        observe_w = dde.PointSetBC(observe_train, w_train, component=1)
        input_data = [bc_a, ic_1, observe_v, observe_w]
    
    if args.dim == 1:
        ## Select relevant PDE
        pde = dynamics.pde_1D
        ## Define the Network
        net = dde.maps.FNN([2] + [hidden_layer_size_1d] * num_hidden_layers_1d + [2], "tanh", "Glorot uniform")
    elif args.dim == 2:
        ## Select relevant PDE
        pde = dynamics.pde_2D
        ## Define the Network
        net = dde.maps.FNN([3] + [hidden_layer_size_2d] * num_hidden_layers_2d + [2], "tanh", "Glorot uniform") 
    data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain, 
                            num_boundary=num_boundary, 
                            anchors=observe_train,
                            num_test=num_test)    
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    ## Stabalize initialization process
    losshistory, _ = model.train(epochs=1)
    num_itr = len(losshistory.loss_train)
    init_loss = max(losshistory.loss_train[num_itr-1])
    num_init = 0
    while init_loss>MAX_LOSS or np.isnan(init_loss):
        num_init += 1
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        losshistory, _ = model.train(epochs=1)
        num_itr = len(losshistory.loss_train)
        init_loss = max(losshistory.loss_train[num_itr-1])
        if num_init > MAX_MODEL_INIT:
            raise ValueError('Model initialization phases exceeded the allowed limit')
            
    ## Train Network
    out_path = dir_path + args.model_folder_name
    if not args.inverse:
        losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path)
    else:
        variables_file = "variables_" + args.inverse + ".dat"
        variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)    
        losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path, callbacks=[variable])
    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    ## Compute rMSE
    model_pred = model.predict(observe_test)
    v_pred = model_pred[:,0:1]
    rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
    print('--------------------------')
    print("V rMSE test:", rmse_v)
    
    # Plot
    data_list = [observe_x, observe_train, V]
    if args.plot and args.dim == 1:
        plot_1d(data_list, model, args.model_folder_name)
    return train_state, v_pred, v_test 

## Run main code
train_state, v_pred, v_test = main(args)