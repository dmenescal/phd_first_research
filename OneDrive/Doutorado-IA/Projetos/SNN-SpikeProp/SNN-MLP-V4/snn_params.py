import numpy as np

def set_default_params():
    
    ## Definition of a set of parameters that will be used for analysis and code recycling too.
    input_params = {}
    
    # Image Patch size
    input_params['sample_hsize'] = 12
    input_params['sample_vsize'] = 10
    input_params['patch_hsize'] = 5
    input_params['patch_vsize'] = 5
    input_params['stride'] = 1
    input_params['patches_set'] = int((input_params['sample_hsize'] - input_params['patch_hsize'])*(input_params['sample_vsize'] - input_params['patch_vsize'])*(1/input_params['stride']))

    # Time duration for spikes generation on LIF neurons
    input_params['lif_simulation_time'] = 80

    # Biological parameters for Encoding and MLP Lif Neurons
    input_params['lif_enc_resistance'] = 10
    input_params['lif_enc_tao'] = 10
    input_params['lif_enc_threshold'] = -50
    input_params['lif_enc_rest_potential'] = -70
    input_params['lif_enc_start_potential'] = -65
    input_params['lif_enc_spike_current'] = 1*(input_params['lif_enc_threshold']-input_params['lif_enc_rest_potential'])/input_params['lif_enc_resistance']

    input_params['lif_mlp_resistance'] = 10
    input_params['lif_mlp_tao'] = 10
    input_params['lif_mlp_threshold'] = -50
    input_params['lif_mlp_rest_potential'] = -70
    input_params['lif_mlp_start_potential'] = -65
    input_params['lif_mlp_spike_current'] = 1*(input_params['lif_mlp_threshold']-input_params['lif_mlp_rest_potential'])/input_params['lif_mlp_resistance']

    # Output size
    input_params['num_classes'] = 8

    # Backpropagation hyperparameters
    input_params['learning_rate'] = 0.001
    input_params['num_epochs'] = 10
    input_params['num_samples'] = 200
    input_params['num_test_samples'] = 0

    # Gradient Calculation variables
    input_params['output_local_gradients'] = np.zeros((input_params['num_classes'], input_params['lif_simulation_time']))
    input_params['encoding_local_gradients'] = np.zeros((input_params['patches_set']))
    
    return input_params
    
def set_param(param_name, param_value):
    input_params[param_name] = param_value