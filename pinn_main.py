import sys
import os         
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import deepxde as dde # version 0.11
from create_plots_1d import plot_1d
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
    parser.add_argument('-ht', '--heter', dest='heter', required = False, action='store_true', help='Predict heterogeneity - only in 2D')    
    args = parser.parse_args()

## Network Parameters
# 1D
input_1d = 2 # network input size (1D)
num_hidden_layers_1d = 4 # number of hidden layers for NN (1D)
hidden_layer_size_1d = 32 # size of each hidden layers (1D)
output_1d = 2 # network input size (1D)
# 2D
input_2d = 3 # network input size (2D)
num_hidden_layers_2d = 4 # number of hidden layers for NN (2D)
hidden_layer_size_2d = 32 # size of each hidden layers (2D)
output_2d = 2 # network output size (2D)
output_heter = 3 # network output size for heterogeneity case (2D)

## Training Parameters
num_domain = 20000 # number of training points within the domain
num_boundary = 1000 # number of training boundary condition points on the geometry boundary
num_test = 1000 # number of testing points within the domain
MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
MAX_LOSS = 4 # upper limit to the initialized loss
epochs = 50000 # number of epochs for training
lr = 0.0005 # learning rate
noise = 0.1 # noise factor
test_size = 0.9 # precentage of testing data

def main(args):
    
    ## Get Dynamics Class
    dynamics = utils.system_dynamics()
    
    ## Parameters to inverse (if needed)
    params = dynamics.params_to_inverse(args.inverse)
    
    ## Generate Data 
    file_name = args.file_name
    observe_x, V, W = dynamics.generate_data(file_name, args.dim)  
    
    ## Split data to train and test
    observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=test_size)
    
    ## Add noise to training data if needed
    if args.noise:
        v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

    ## Geometry and Time domains
    geomtime = dynamics.geometry_time(args.dim)
    ## Define Boundary Conditions
    bc_1 = dynamics.BC_func(args.dim, geomtime)
    ## Define Initial Conditions
    ic_1 = dynamics.IC_func(observe_train, v_train)
    
    ## Model observed data
    observe_v = dde.PointSetBC(observe_train, v_train, component=0)
    input_data = [bc_1, ic_1, observe_v]
    if args.w_input: ## If W required as input
        observe_w = dde.PointSetBC(observe_train, w_train, component=1)
        input_data = [bc_1, ic_1, observe_v, observe_w]
    
    ## Select relevant PDE (Dim, Heterogeneity) and define the Network 
    if args.dim == 1:
        pde = dynamics.pde_1D
        # net = dde.maps.ResNet(2, 2, 32, 6, "tanh", kernel_initializer="Glorot uniform")
        net = dde.maps.FNN([input_1d] + [hidden_layer_size_1d] * num_hidden_layers_1d + [output_1d], "tanh", "Glorot uniform")
    elif args.dim == 2 and args.heter:
        pde = dynamics.pde_2D_heter    
        net = dde.maps.FNN([input_2d] + [hidden_layer_size_2d] * num_hidden_layers_2d + [output_heter], "tanh", "Glorot uniform") 
        net.apply_output_transform(dynamics.modify_output_heter)
    elif args.dim == 2 and not args.heter:
        pde = dynamics.pde_2D    
        net = dde.maps.FNN([input_2d] + [hidden_layer_size_2d] * num_hidden_layers_2d + [output_2d], "tanh", "Glorot uniform") 
    data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain, 
                            num_boundary=num_boundary, 
                            anchors=observe_train,
                            num_test=num_test)    
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    ## Stabalize initialization process
    losshistory, _ = model.train(epochs=1)
    initial_loss = max(losshistory.loss_train[0])
    num_init = 0
    while initial_loss>MAX_LOSS or np.isnan(initial_loss):
        num_init += 1
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        losshistory, _ = model.train(epochs=1)
        initial_loss = max(losshistory.loss_train[0])
        if num_init > MAX_MODEL_INIT:
            raise ValueError('Model initialization phases exceeded the allowed limit')
            
    ## Train Network
    out_path = dir_path + args.model_folder_name
    if args.inverse:
        variables_file = "variables_" + args.inverse + ".dat"
        variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)    
        losshistory, train_state = model.train(epochs=epochs-1, model_save_path = out_path, callbacks=[variable])
    else:
        losshistory, train_state = model.train(epochs=epochs-1, model_save_path = out_path)
        
    ## Compute rMSE
    model_pred = model.predict(observe_test)
    v_pred = model_pred[:,0:1]   
    rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
    print('--------------------------')
    print("V rMSE for test data:", rmse_v)
    print('--------------------------')
    print("Arguments: ", args)
    
    # Plot
    data_list = [observe_x, observe_train, v_train, V]
    if args.plot and args.dim == 1:
        plot_1d(data_list,dynamics, model, args.model_folder_name)
    return model

## Run main code
model = main(args)
