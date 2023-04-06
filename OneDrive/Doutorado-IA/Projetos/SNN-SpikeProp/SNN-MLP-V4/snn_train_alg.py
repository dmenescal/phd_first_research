import numpy as np


###########################################################################################
## Mapping Function. It Applies the input into the neural net and returns the output.
def map(input_sample, encoding_weights, mlp_weights):
    patches = patches_extraction(input_sample, patch_size=input_params['patch_hsize'], stride=input_params['stride'])
    encoding_current = encoding_synaptic_current(patches, encoding_weights)
    encoding_st, encoding_st_times, encoding_st_count = spikes_generation(layer_name='encoding', input_current=encoding_current,
                                                                          neuron_resistance=input_params['lif_enc_resistance'], 
                                                                          time_constant=input_params['lif_enc_tao'], 
                                                                          rest_potential=input_params['lif_enc_rest_potential'], 
                                                                          start_potential=input_params['lif_enc_start_potential'],
                                                                          threshold=input_params['lif_enc_threshold'], 
                                                                          simulation_time=input_params['lif_simulation_time'])
    #print("Encoding Spikes:")
    #print(encoding_st_times)
    
    mlp_current = mlp_synaptic_current(encoding_st, mlp_weights)
    
    #print("Corrente de Saida (MLP):")
    #print(mlp_current.shape)
    #print(mlp_current)
    mlp_st, mlp_st_times, mlp_st_count = spikes_generation(layer_name='mlp', input_current=mlp_current,
                                                                          neuron_resistance=input_params['lif_mlp_resistance'], 
                                                                          time_constant=input_params['lif_mlp_tao'], 
                                                                          rest_potential=input_params['lif_mlp_rest_potential'], 
                                                                          start_potential=input_params['lif_mlp_start_potential'],
                                                                          threshold=input_params['lif_mlp_threshold'], 
                                                                          simulation_time=input_params['lif_simulation_time'])
    #print("Output Spikes:")
    #print(mlp_st_times.shape)
    #print(mlp_st_times)
    return mlp_st, mlp_st_count, mlp_st_times, encoding_current, mlp_current

###########################################################################################
## Forward Pass Main Function. It maps one sample from the set into the error_signal and 
## error energy.
def forwardPass(input_sample, input_sample_label, encoding_weights, mlp_weights):
    mlp_st, mlp_spk_count, mlp_st_times, encoding_current, mlp_current = map(input_sample, encoding_weights, mlp_weights)
    
    error_signal = dissimilarity_error(mlp_st_times[input_sample_label],input_params['desired_times'][input_sample_label])
    return error_signal, encoding_current, mlp_current

###########################################################################################
## MLP Local Gradient Evaluation
def mlp_local_gradient(sample_error_signal, mlp_current):
    local_gradient = np.zeros((input_params['num_classes'], input_params['lif_simulation_time']))
    for k in range(input_params['num_classes']):
        for t in range(input_params['lif_simulation_time']):
            local_gradient[k][t] = (-1)*sample_error_signal[t]*lif_activation_derivative(R     = input_params['lif_mlp_resistance'],
                                                                                        I     = mlp_current[k], 
                                                                                        V_rest= input_params['lif_mlp_rest_potential'], 
                                                                                        V0    = input_params['lif_mlp_start_potential'], 
                                                                                        tao   = input_params['lif_mlp_tao'], 
                                                                                        t     = t)
    return local_gradient

###########################################################################################
## Encoding Local Gradient Evaluation
def enc_local_gradient(mlp_gradients, mlp_weights, encoding_current):
    local_gradient = np.zeros((input_params['patches_set']))
    for e in range(input_params['patches_set']):
        acc = 0
        for k in range(input_params['num_classes']):
            for t in range(input_params['lif_simulation_time']):
                acc += mlp_gradients[k][t]*mlp_weights[k][e][t]
        local_gradient[e] = acc*lif_activation_derivative(R      = input_params['lif_enc_resistance'],
                                                          I      = encoding_current[e], 
                                                          V_rest = input_params['lif_enc_rest_potential'], 
                                                          V0     = input_params['lif_enc_start_potential'], 
                                                          tao    = input_params['lif_enc_tao'], 
                                                          t      = t)
        
    return local_gradient

###########################################################################################
## Delta Rule for MLP Layer
def mlp_delta_rule(mlp_gradient, mlp_output):
    return input_params['learning_rate']*mlp_gradient*mlp_output

###########################################################################################
## Delta Rule for Encoding Layer
def encoding_delta_rule(encoding_gradient, encoding_output):
    delta = []
    for d in range(input_params['patches_set']):
        delta.append(input_params['learning_rate']*encoding_gradient[d]*encoding_output[d])
    return np.asarray(delta)

###########################################################################################
## Compute New Weights for MLP Layer
def compute_new_mlp_weights(mlp_weights, delta):
    new_weights = []
    for d in range(input_params['patches_set']):
        new_weights.append(mlp_weights[:, d]+delta)
    return np.reshape(np.asarray(new_weights), (input_params['num_classes'], input_params['patches_set'], input_params['lif_simulation_time']))

###########################################################################################
## Compute New Weights for Encoding Layer
def compute_new_enc_weights(enc_weights, delta):
    new_weights = []
    for d in range(input_params['patches_set']):
        new_weights.append(enc_weights[d]+delta[d])
    return np.asarray(new_weights)

###########################################################################################
## Backpropagation function. It uses the error signal and energy to provide new synaptic
## weights adjustments.
def spike_backpropagation(error_signal, encoding_current, mlp_current):
    mlp_local_grad = mlp_local_gradient(error_signal, mlp_current)
    enc_local_grad = enc_local_gradient(mlp_local_grad, input_params['mlp_weights'], encoding_current)
    mlp_delta = mlp_delta_rule(mlp_local_grad, input_params['mlp_spikes'])
    enc_delta = encoding_delta_rule(enc_local_grad, input_params['encoding_spike_count'])
 
    new_mlp_W = compute_new_mlp_weights(input_params['mlp_weights'], mlp_delta)
    new_enc_W = compute_new_enc_weights(input_params['encoding_weights'], enc_delta)
    
    return new_mlp_W, new_enc_W

###########################################################################################
## Fitting function
def fit(sample_set):
    W_enc = np.random.uniform(0, 1, (input_params['patches_set'], input_params['patch_hsize'], input_params['patch_vsize']))
    input_params['encoding_weights'] = W_enc

    W_mlp = np.random.uniform(0, 1, (input_params['num_classes'], input_params['patches_set'], input_params['lif_simulation_time']))
    input_params['mlp_weights'] = W_mlp

    training_error = np.zeros(input_params['num_epochs'])
    
    for epoch_count in range(input_params['num_epochs']):
        sse_sum = 0
        for sample_count in range(input_params['num_samples']):
            err_sum = 0
            for class_count in range(input_params['num_classes']):
                #print("Epoca {} // Amostra {} // Classe {}".format(epoch_count, sample_count, class_count))
                error_signal, I_enc, I_mlp = forwardPass(input_sample=sample_set[class_count, sample_count], 
                                                            input_sample_label=class_count,
                                                            encoding_weights=input_params['encoding_weights'], 
                                                            mlp_weights=input_params['mlp_weights'])
                mw, ew = spike_backpropagation(error_signal, I_enc, I_mlp)
                input_params['mlp_weights'] = mw
                input_params['encoding_weights'] = ew
                err_sum += quadratic_error(error_signal)
            #num_cores = multiprocessing.cpu_count()
            #weights = Parallel(n_jobs=16)(delayed(forward_backward)(c, sample_count, sample_set, err_sum) for c in range(input_params['num_classes']))
            sse_sum += 0.5*err_sum
        mse = sse_sum/(input_params['patches_set']*input_params['num_classes']*input_params['num_samples'])
        print("MSE - Epoch {}:{}".format(epoch_count, mse))
        training_error[epoch_count] = mse
    return training_error