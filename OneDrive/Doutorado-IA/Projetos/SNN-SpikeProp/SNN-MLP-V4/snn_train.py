## Libraries used
import math
import numpy as np
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from sklearn import preprocessing

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
input_params['lif_simulation_time'] = 100

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
input_params['num_classes'] = 10

# Backpropagation hyperparameters
input_params['learning_rate'] = 0.01
input_params['num_epochs'] = 1
input_params['num_samples'] = 10
input_params['num_test_samples'] = 0

# Gradient Calculation variables
input_params['output_local_gradients'] = np.zeros((input_params['num_classes'], input_params['lif_simulation_time']))
input_params['encoding_local_gradients'] = np.zeros((input_params['patches_set']))


###########################################################################################
## Setter for Input Params
def set_input_params(param_name, param_value):
    input_params[param_name] = param_value


###########################################################################################
## Getter for Input Params Initial Configuration
def get_input_params():
    return input_params

###########################################################################################
# A geração dos spike trains esperados para a rede neural, a priori, não é definido por um conjunto de valores fixos.
# Cada classe é representada por um array de 100 posições com valores inteiros, representando cada momento em que um spike foi gerado ou não.
# Para não criar um viés na definição e deixar o mais "randômico" possível, para cada classe esperada (0 a 9), serão geradas 
# sequências de valores aleatórios dentro de intervalos definidos para cada classe:
# Classe 0: Intervalo de valores inteiros de 0 a 9
# Classe 1: Intervalo de valores inteiros de 10 a 19
# Classe 2: Intervalo de valores inteiros de 20 a 29
# Classe 3: Intervalo de valores inteiros de 30 a 39
# Classe 4: Intervalo de valores inteiros de 40 a 49
# Classe 5: Intervalo de valores inteiros de 50 a 59
# Classe 6: Intervalo de valores inteiros de 60 a 69
# Classe 7: Intervalo de valores inteiros de 70 a 79
# Classe 8: Intervalo de valores inteiros de 80 a 89
# Classe 9: Intervalo de valores inteiros de 90 a 99
# Dessa forma a união dos intervalos vai corresponder ao intervalo completo de simulação, que no caso é de 100 passos ("ms")
def spike_interval_generator(number_of_classes=10, simulation_time=100, class_idx=-1):
    interval = []
    for i in range(number_of_classes):
        interval.append((i*int((simulation_time/number_of_classes)),-1 + ((i+1)*int((simulation_time/number_of_classes)))))
        #interval.append((0, simulation_time-1))
    if (class_idx >= 0): #Retorna apenas o intervalo correspondente da classe escolhida
        return interval[class_idx]
    
    return interval

###########################################################################################
## Interspike time (ISI) Generator for Desired spikes
def spike_isi_generator(number_of_classes=10, simulation_time=100):
    spike_isi_per_class = np.zeros(number_of_classes)
    
    for i in range(number_of_classes):
        spike_isi_per_class[i] = np.random.randint(1, int((simulation_time/number_of_classes)))
        
    return spike_isi_per_class

###########################################################################################
## Spike Count Generator for Desired spikes
def spike_count_generator(number_of_classes=10, simulation_time=100, class_idx=-1):
    spike_counts_per_class = np.zeros(number_of_classes)
    
    for i in range(number_of_classes):
        spike_counts_per_class[i] = np.random.randint(2, int((simulation_time/number_of_classes)))
        
    if (class_idx >= 0):#Retorna apenas as contagens correspondentes da classe escolhida
        return spike_counts_per_class[class_idx]
    
    return spike_counts_per_class

###########################################################################################
## Spike Times Generator for Desired spikes
def spike_times_generator(spike_range, spike_count, class_idx):
    desired_spike_train = []
    desired_spike_bin = np.zeros((input_params['num_classes'], input_params['lif_simulation_time']))
    num_classes = input_params['num_classes']
    
    for i in range(num_classes):
        spike_time = random.sample(range(spike_range[i][0], spike_range[i][1]), int(spike_count[i]))
        desired_spike_bin[i][spike_time] = 1
        desired_spike_train.append(np.sort(spike_time))
    
    if (class_idx >= 0):
        return  desired_spike_bin[class_idx]
    
    return desired_spike_bin, np.asarray(desired_spike_train)

###########################################################################################
## New Spike Times Generator for Desired spikes
def spike_times_generator_new(spike_range, spike_isi):
    desired_spike_train = []
    desired_spike_bin = np.zeros((input_params['num_classes'], input_params['lif_simulation_time']))
    num_classes = input_params['num_classes']
    for i in range(num_classes):
        spike_times = []
        dt = spike_range[i][1] - spike_range[i][0] + 1
        spike_count = int(dt/spike_isi[i])
        
        #print("Range da Classe: ", dt)
        #print("ISI da Classe: ", spike_isi[i])
        #print("Spike Count da Classe: ", spike_count)
        
        for c in range(spike_count):
            spike_times.append(int(spike_range[i][0]+(c*spike_isi[i])))
        
        #print("Classe "+str(i)+ ":", spike_times)
        desired_spike_bin[i][spike_times] = 1
        spike_times = np.array(spike_times)
        desired_spike_train.append(np.sort(spike_times))

    return desired_spike_bin, np.asarray(desired_spike_train)

###########################################################################################
## Main function to return desired spikes
def get_desired_spikes():
    desired_range = spike_interval_generator(simulation_time=input_params['lif_simulation_time'])
    #desired_range = (0,input_params['lif_simulation_time']-1) # Disconsider different ranges.
    input_params['derired_range'] = desired_range
    desired_count = spike_count_generator(simulation_time=input_params['lif_simulation_time'])
    input_params['desired_count'] = desired_count
    dsp = spike_times_generator(desired_range, desired_count, -1)
    return dsp

###########################################################################################
## Set Maximum/Minimum InputCurrent value according to interspike interval time
def set_current(R, Vth, El, V0, tao, t_isi):
    num = (Vth-El)-(V0-El)*math.exp(t_isi/tao)
    den = R*(1-math.exp(t_isi/tao))
    
    return num/den

def set_max_current():
    input_params['max_enc_input_current'] = set_current(R=input_params['lif_enc_resistance'],
                                                    Vth=input_params['lif_enc_threshold'],
                                                    El=input_params['lif_enc_rest_potential'],
                                                    V0=input_params['lif_enc_start_potential'],
                                                    tao=input_params['lif_enc_tao'],
                                                    t_isi=1)
    input_params['max_mlp_input_current'] = set_current(R=input_params['lif_mlp_resistance'],
                                                    Vth=input_params['lif_mlp_threshold'],
                                                    El=input_params['lif_mlp_rest_potential'],
                                                    V0=input_params['lif_mlp_start_potential'],
                                                    tao=input_params['lif_mlp_tao'],
                                                    t_isi=1)

def set_min_current():
    input_params['min_enc_input_current'] = set_current(R=input_params['lif_enc_resistance'],
                                                    Vth=input_params['lif_enc_threshold'],
                                                    El=input_params['lif_enc_rest_potential'],
                                                    V0=input_params['lif_enc_start_potential'],
                                                    tao=input_params['lif_enc_tao'],
                                                    t_isi=input_params['lif_simulation_time'])
    input_params['min_mlp_input_current'] = set_current(R=input_params['lif_mlp_resistance'],
                                                    Vth=input_params['lif_mlp_threshold'],
                                                    El=input_params['lif_mlp_rest_potential'],
                                                    V0=input_params['lif_mlp_start_potential'],
                                                    tao=input_params['lif_mlp_tao'],
                                                    t_isi=input_params['lif_simulation_time'])
    
    
###########################################################################################
## Image patches extraction
def patches_extraction(image, stride, patch_size):
    patches = []
    num_patches = 0
    ci = 0
    hn = image.shape[0]
    vn = image.shape[1]
    while(ci < hn):
        cj = 0
        while(cj < vn):
            pivot = (ci, cj)
            if(pivot[0]+patch_size >= hn or pivot[1]+patch_size >= vn):
                cj += stride
            else:
                patch = image[pivot[0]:pivot[0]+patch_size, pivot[1]:pivot[1]+patch_size]
                patches.append(patch)
                num_patches += 1
                cj += stride
        ci += stride
    #input_params['encoding_layer_neurons_num'] = num_patches
    return np.asarray(patches)

###########################################################################################
## Encoding Synaptic Current
def encoding_synaptic_current(patches, encoding_weights):
    num_patches = input_params['patches_set']
    I_enc = np.zeros(num_patches)
    for p in range(num_patches):
        acc = 0
        for i in range(input_params['patch_hsize']):
            for j in range(input_params['patch_vsize']):
                acc += (patches[p][i][j]*encoding_weights[p][i][j])
        I_enc[p] = acc
    #print("Encoding Current Before Normalization:", I_enc)
    I_enc_norm = (I_enc-input_params['min_enc_input_current'])/(input_params['max_enc_input_current']-input_params['min_enc_input_current'])
    #print("Encoding Current After Normalization:", I_enc_norm)
    input_params['encoding_current'] = I_enc_norm
    #print("MLP CURRENT: ", I_enc_norm)
    return I_enc_norm


###########################################################################################
## MLP Synaptic Current
def mlp_synaptic_current(encoding_spike_train, mlp_weights):
    synaptic_current = []
    for neuron in range(input_params['num_classes']):
        current = 0
        for u in range(input_params['patches_set']):
            #print("Encoding Spike Trains: ", encoding_spike_train[u])
            for s in range(input_params['lif_simulation_time']):
                current += (encoding_spike_train[u][s] * mlp_weights[neuron][u][s]) 
        synaptic_current.append(current)
    I_mlp = np.asarray(synaptic_current)
    #print("Mlp Current Before Normalization:", I_mlp)
    I_mlp_norm = (I_mlp - input_params['min_mlp_input_current'])/(input_params['max_mlp_input_current']-input_params['min_mlp_input_current'])
    input_params['mlp_current'] = I_mlp_norm
    #print("Mlp Current After Normalization:", I_mlp_norm)
    #print("MLP CURRENT: ", I_mlp_norm)
    return I_mlp_norm

###########################################################################################
## New MLP Synaptic Current
def mlp_synaptic_current_new(encoding_spike_train, mlp_weights):
    synaptic_current = []
    for out_neuron in range(input_params['num_classes']):
        current = 0
        for enc_neuron in range(input_params['patches_set']):
            for s in range(input_params['lif_simulation_time']):
                current += (encoding_spike_train[enc_neuron][out_neuron][s] * mlp_weights[out_neuron][enc_neuron][s])#/input_params['lif_mlp_spike_current']
        synaptic_current.append(current)
    I_mlp = np.asarray(synaptic_current)
    I_mlp_norm = (I_mlp - input_params['min_mlp_input_current'])/(input_params['max_mlp_input_current']-input_params['min_mlp_input_current'])
    input_params['mlp_current'] = I_mlp_norm
    #print("MLP CURRENT: ", I_mlp_norm)
    return I_mlp_norm

###########################################################################################
## LIF Neuron Potential Dynamics
def lif_activation(R, I, V_rest, V0, tao, t):
    return (V_rest+(R*I))+((V0-V_rest-(R*I))*math.exp((-t)/(tao)))#(V_rest+(R*I))+(V0-V_rest-(R*I))*math.exp((-1*t)/(tao))

def integrate_and_fire_spikes(lif_R, lif_tao, lif_rest_voltage, lif_start_voltage, lif_threshold, I_syn, time_duration):
    spike_train = []
    spike_time = []
    spike_count = 0
    lif_timer = 0
    
    for t in range(time_duration):
        new_voltage = lif_activation(lif_R, I_syn, lif_rest_voltage, lif_start_voltage, lif_tao, lif_timer)
        if (new_voltage >= lif_threshold):
            spike_train.append(1)
            spike_time.append(t)
            spike_count += 1
            lif_timer = 0
        else:
            spike_train.append(0)
            lif_timer += 1
            
    return np.asarray(spike_train), np.asarray(spike_time), spike_count

###########################################################################################
## Spikes Generation - V0
def spikes_generation(layer_name, input_current, neuron_resistance, time_constant, rest_potential, start_potential, threshold, simulation_time):
    st = []
    st_times = []
    st_count = []

    if(layer_name == 'mlp'):
        iteration_set = input_params['num_classes']
        spike_train_param = 'mlp_spikes'
        spike_time_param  = 'mlp_spike_times'
        spike_count_param = 'mlp_spike_count'
    else: # 'encoding'
        iteration_set = input_params['patches_set']
        spike_train_param = 'encoding_spikes'
        spike_time_param  = 'encoding_spike_times'
        spike_count_param = 'encoding_spike_count'
    
    for neuron in range(iteration_set):
        spikes, spike_times, spike_count = integrate_and_fire_spikes(lif_R=neuron_resistance, 
                                                                        lif_tao=time_constant,
                                                                        lif_rest_voltage=rest_potential, 
                                                                        lif_start_voltage=start_potential, 
                                                                        lif_threshold=threshold, 
                                                                        I_syn=input_current[neuron], 
                                                                        time_duration=simulation_time)
        st.append(spikes)
        st_times.append(spike_times)
        st_count.append(spike_count)

    st = np.asarray(st)
    st_times = np.asarray(st_times)
    st_count = np.asarray(st_count)

    input_params[spike_train_param] = st
    input_params[spike_time_param] = st_times
    input_params[spike_count_param] = st_count
    
    return st, st_times, st_count


###########################################################################################
## Encoding Layer Spikes Generation - V1
def encoding_layer_spikes_generation(input_current, neuron_resistance, time_constant, 
                                     rest_potential, start_potential, threshold, simulation_time):

    spike_train_param = 'encoding_spikes'
    spike_time_param  = 'encoding_spike_times'
    spike_count_param = 'encoding_spike_count'
    

    st_times_enc = []
    st_bin_enc = []
    st_count_enc = []
    for enc_neuron in range(input_params['patches_set']):
        st_times_out = []
        st_bin_out = []
        st_count_out = []
        for out_neuron in range(input_params['num_classes']):
            spikes, spike_times, spike_count = integrate_and_fire_spikes(lif_R=neuron_resistance, 
                                                                            lif_tao=time_constant,
                                                                            lif_rest_voltage=rest_potential, 
                                                                            lif_start_voltage=start_potential, 
                                                                            lif_threshold=threshold, 
                                                                            I_syn=input_current[enc_neuron], 
                                                                            time_duration=simulation_time)
            st_times_out.append(spike_times)
            st_bin_out.append(spikes)
            st_count_out.append(spike_count)
        st_times_enc.append(st_times_out)
        st_bin_enc.append(st_bin_out)
        st_count_enc.append(st_count_out)

    st = np.asarray(st_bin_enc)
    st_times = np.asarray(st_times_enc)
    st_count = np.asarray(st_count_enc)

    input_params[spike_train_param] = st
    input_params[spike_time_param] = st_times
    input_params[spike_count_param] = st_count
    return st, st_times, st_count


###########################################################################################
## Output Layer Spikes Generation - V1 : Nao itera sobre todos os neuronios da camada, apenas o desejado.
def output_layer_spikes_generation(input_current, output_neuron_label, neuron_resistance, 
                                   time_constant, rest_potential, start_potential, threshold, simulation_time):
    
    spike_bin, spike_times, spike_count = integrate_and_fire_spikes(lif_R=neuron_resistance, 
                                                                        lif_tao=time_constant,
                                                                        lif_rest_voltage=rest_potential, 
                                                                        lif_start_voltage=start_potential, 
                                                                        lif_threshold=threshold, 
                                                                        I_syn=input_current[output_neuron_label], 
                                                                        time_duration=simulation_time)
    
    spike_train_param = 'mlp_spikes'
    spike_time_param  = 'mlp_spike_times'
    spike_count_param = 'mlp_spike_count'
    
    input_params[spike_train_param] = spike_bin
    input_params[spike_time_param] = spike_times
    input_params[spike_count_param] = spike_count
    print("MLP SPIKE TRAIN: ", spike_times)
    return spike_bin, spike_times, spike_count



###########################################################################################
## Error signal Evaluation
def evaluate_error_signal(desired_spikes, model_spikes):
    return desired_spikes - model_spikes

###########################################################################################
## Instantaneous error energy value
def error_energy_value(error_signal):
    sqr_error = 0
    for k in range(input_params['num_classes']):
        for t in range(input_params['lif_simulation_time']):
            sqr_error += (error_signal[k][t])**2
    return 0.5*sqr_error

def quadratic_error(error_signal):
    sqr_error = 0
    for t in range(input_params['lif_simulation_time']):
        sqr_error += (error_signal[t])**2
    return sqr_error

###########################################################################################
## New Mapping Function. It Applies the input into the neural net and returns the output for a specific output neuron.
def map_new(input_sample, input_sample_label, encoding_weights, mlp_weights):
    patches = patches_extraction(input_sample, patch_size=input_params['patch_hsize'], stride=input_params['stride'])
    encoding_current = encoding_synaptic_current(patches, encoding_weights)
    #print("Corrente de Encoding:")
    #print(encoding_current.shape)
    #print(encoding_current)
    encoding_st, encoding_st_times, encoding_st_count = encoding_layer_spikes_generation(input_current=encoding_current,
                                                                                        neuron_resistance=input_params['lif_enc_resistance'], 
                                                                                        time_constant=input_params['lif_enc_tao'], 
                                                                                        rest_potential=input_params['lif_enc_rest_potential'], 
                                                                                        start_potential=input_params['lif_enc_start_potential'],
                                                                                        threshold=input_params['lif_enc_threshold'], 
                                                                                        simulation_time=input_params['lif_simulation_time'])
    
    mlp_current = mlp_synaptic_current_new(encoding_st, mlp_weights)
    
    mlp_st, mlp_st_times, mlp_st_count = output_layer_spikes_generation(input_current=mlp_current, output_neuron_label=input_sample_label,
                                                                          neuron_resistance=input_params['lif_mlp_resistance'], 
                                                                          time_constant=input_params['lif_mlp_tao'], 
                                                                          rest_potential=input_params['lif_mlp_rest_potential'], 
                                                                          start_potential=input_params['lif_mlp_start_potential'],
                                                                          threshold=input_params['lif_mlp_threshold'], 
                                                                          simulation_time=input_params['lif_simulation_time'])
    return mlp_st, mlp_st_count, mlp_st_times, encoding_current, mlp_current


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
## New Forward Pass Main Function. It maps one sample from the set into the error_signal and 
## error energy.
def forwardPass_new(input_sample, input_sample_label, encoding_weights, mlp_weights):
    output_st, output_spk_count, output_st_times, encoding_current, mlp_current = map_new(input_sample, input_sample_label, encoding_weights, mlp_weights)
    error_signal = dissimilarity_error(output_st_times,input_params['desired_times'][input_sample_label])
    return error_signal, encoding_current, mlp_current

###########################################################################################
## LIF Activation Derivative
def lif_activation_derivative(R, I, V_rest, V0, tao, t):
    return ((V_rest+(R*I)-V0)*math.exp((-1*t)/(tao)))/tao  

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
## New MLP Local Gradient Evaluation
def mlp_local_gradient_new(sample_label, sample_error_signal, mlp_current):
    for t in range(input_params['lif_simulation_time']):
        input_params['output_local_gradients'][sample_label][t] = (-1)*sample_error_signal[t]*lif_activation_derivative(R     = input_params['lif_mlp_resistance'],
                                                                                                                        I     = mlp_current[sample_label], 
                                                                                                                        V_rest= input_params['lif_mlp_rest_potential'], 
                                                                                                                        V0    = input_params['lif_mlp_start_potential'], 
                                                                                                                        tao   = input_params['lif_mlp_tao'], 
                                                                                                                        t     = t)

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
## New Encoding Local Gradient Evaluation
def enc_local_gradient_new(sample_label, mlp_weights, encoding_current):
    for e in range(input_params['patches_set']):
        acc = 0
        for t in range(input_params['lif_simulation_time']):
            acc += input_params['output_local_gradients'][sample_label][t]*mlp_weights[sample_label][e][t]
        input_params['encoding_local_gradients'][e] = acc*lif_activation_derivative(R      = input_params['lif_enc_resistance'],
                                                                                    I      = encoding_current[e], 
                                                                                    V_rest = input_params['lif_enc_rest_potential'], 
                                                                                    V0     = input_params['lif_enc_start_potential'], 
                                                                                    tao    = input_params['lif_enc_tao'], 
                                                                                    t      = t)

###########################################################################################
## Delta Rule for MLP Layer
def mlp_delta_rule(mlp_gradient, mlp_output):
    return input_params['learning_rate']*mlp_gradient*mlp_output

###########################################################################################
## New Delta Rule for MLP Layer
def mlp_delta_rule_new(sample_label, mlp_neuron_output):
    return input_params['learning_rate']*input_params['output_local_gradients'][sample_label]*mlp_neuron_output

###########################################################################################
## Delta Rule for Encoding Layer
def encoding_delta_rule(encoding_gradient, encoding_output):
    delta = []
    for d in range(input_params['patches_set']):
        delta.append(input_params['learning_rate']*encoding_gradient[d]*encoding_output[d])
    return np.asarray(delta)

###########################################################################################
## New Delta Rule for Encoding Layer
def encoding_delta_rule_new(encoding_output):
    delta = []
    for d in range(input_params['patches_set']):
        delta.append(input_params['learning_rate']*input_params['encoding_local_gradients'][d]*encoding_output[d])
    return np.asarray(delta)

###########################################################################################
## Compute New Weights for MLP Layer
def compute_new_mlp_weights(mlp_weights, delta):
    new_weights = []
    for d in range(input_params['patches_set']):
        new_weights.append(mlp_weights[:, d]+delta)
    return np.reshape(np.asarray(new_weights), (input_params['num_classes'], input_params['patches_set'], input_params['lif_simulation_time']))

###########################################################################################
## Compute New Weights for MLP Layer
def compute_new_mlp_weights_v1(sample_label,mlp_weights, delta):
    new_weights = []
    for d in range(input_params['patches_set']):
        new_weights.append(mlp_weights[sample_label, d]+delta)
    return np.reshape(np.asarray(new_weights), (1, input_params['patches_set'], input_params['lif_simulation_time']))

###########################################################################################
## Compute New Weights for Encoding Layer
def compute_new_enc_weights(enc_weights, delta):
    new_weights = []
    for d in range(input_params['patches_set']):
        new_weights.append(enc_weights[d]+delta[d])
    return np.asarray(new_weights)

###########################################################################################
## MaxMin(0,1) function. It normalizes an array into a (0,1) interval.
def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

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

    #norm_mlp_W = normalize(new_mlp_W)
    #norm_enc_W = normalize(new_enc_W)
    
    return new_mlp_W, new_enc_W

###########################################################################################
## Backpropagation function. It uses the error signal and energy to provide new synaptic
## weights adjustments.
def spike_backpropagation_new(error_signal, encoding_current, mlp_current, sample_label):
    mlp_local_gradient_new(sample_label, error_signal, mlp_current)
    enc_local_gradient_new(sample_label, input_params['mlp_weights'], encoding_current)
    
    mlp_delta = mlp_delta_rule_new(sample_label, input_params['mlp_spikes'])
    enc_delta = encoding_delta_rule_new(input_params['encoding_spike_count'])
    
    print("MLP GRADIENT: ", input_params['output_local_gradients'][sample_label])
    print("MLP DELTA: ", mlp_delta)
    print("ENC GRADIENT: ", input_params['encoding_local_gradients'])
    print("ENC DELTA: ", enc_delta)
    
    new_mlp_W = compute_new_mlp_weights_v1(sample_label, input_params['mlp_weights'], mlp_delta)
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
                error_signal, I_enc, I_mlp = forwardPass_new(input_sample=sample_set[class_count, sample_count], 
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

###########################################################################################
## New Fitting function
def fit_new(sample_set):
    W_enc = np.random.uniform(0, 1, (input_params['patches_set'], input_params['patch_hsize'], input_params['patch_vsize']))
    input_params['encoding_weights'] = W_enc

    W_mlp = np.random.uniform(0, 1, (input_params['num_classes'], input_params['patches_set'], input_params['lif_simulation_time']))
    input_params['mlp_weights'] = W_mlp

    #print(W_mlp)
    
    training_error = np.zeros(input_params['num_epochs'])
    
    for epoch_count in range(input_params['num_epochs']):
        sse_sum = 0
        for sample_count in range(input_params['num_samples']):
            err_sum = 0
            for class_count in range(input_params['num_classes']):
                print("####################################################################################")
                print("EPOCH {}, SAMPLE {}, LABEL/CLASS {}".format(epoch_count, sample_count, class_count))
                #print("ENCODING WEIGHTS: ", input_params['encoding_weights'])
                #print("MLP WEIGHTS: ", input_params['mlp_weights'])
                error_signal, I_enc, I_mlp = forwardPass_new(input_sample=sample_set[class_count, sample_count], 
                                                             input_sample_label=class_count,
                                                             encoding_weights=input_params['encoding_weights'], 
                                                             mlp_weights=input_params['mlp_weights'])
                #print(error_signal)
                mw, ew = spike_backpropagation_new(error_signal, I_enc, I_mlp, class_count)
                #print(mw)
                input_params['mlp_weights'][class_count] = mw
                input_params['encoding_weights'] = ew
                err_sum += quadratic_error(error_signal)
                print("####################################################################################")
            sse_sum += 0.5*err_sum
        mse = sse_sum/(input_params['patches_set']*input_params['num_classes']*input_params['num_samples'])
        print("MSE - Epoch {}:{}".format(epoch_count, mse))
        training_error[epoch_count] = mse
    return training_error

###########################################################################################
## Image Denoise function
def img_noise(sample, prob):
    noisy_img = np.zeros(sample.shape)
    noisy_samples = np.zeros(sample.shape)
    threshold = 1 - prob
    for i in range(sample.shape[0]):
      for j in range(sample.shape[1]):
        rdn = random.random()
        if (rdn < prob):
            noisy_img[i][j] = 0
        elif (rdn > threshold):
            noisy_img[i][j] = 1
            noisy_samples[i][j] = 1
        else:
            noisy_img[i][j] = sample[i][j]
    return noisy_img

###########################################################################################
# Tempos do spike antecessor ao instante t
def tP(t, neuron_spike_times):
    res = neuron_spike_times[np.where(neuron_spike_times <= t)]
    if(res.shape[0] == 0):
        return 0
    else:
        return max(res)

###########################################################################################
# Tempos do spike sucessor ao instante t   
def tF(t, neuron_spike_times):
    res = neuron_spike_times[np.where(neuron_spike_times > t)]
    if(res.shape[0] == 0):
        return 0
    else:
        return min(res)
    
###########################################################################################
# Intervalo entre spikes instantaneo
def x_isi(t, neuron_spike_times):
    return tF(t, neuron_spike_times)-tP(t, neuron_spike_times)

def media_x_isi(t, spike_times):
    return 0.5*(x_isi(t, spike_times[0])+x_isi(t, spike_times[1]))

###########################################################################################
# Diferencas absolutas entre sucessores e antecessores
def delta_tP(t, spike_times):
    return abs(tP(t, spike_times[0])-tP(t, spike_times[1]))

def delta_tF(t, spike_times):
    return abs(tF(t, spike_times[0])-tF(t, spike_times[1]))

###########################################################################################
# Intervalos para os spikes antecessores e sucessores
def xP(t, neuron_spike_times):
    return t - tP(t, neuron_spike_times)

def xF(t, neuron_spike_times):
    return tF(t, neuron_spike_times) - t

def media_xp(t, spike_times):
    return 0.5*(xP(t, spike_times[0]) + xP(t, spike_times[1]))

def media_xf(t, spike_times):
    return 0.5*(xF(t, spike_times[0]) + xF(t, spike_times[1]))

###########################################################################################
# Perfil de dissimiliaridade entre os spike trains - original
def So(t, spike_times):
    numerador = (delta_tP(t, spike_times)*media_xf(t, spike_times))+(delta_tF(t, spike_times)*media_xp(t, spike_times))
    denominador = media_x_isi(t, spike_times)**2
    if (denominador == 0): denominador = 0.0001
    return numerador/denominador

###########################################################################################
# Calcula a Dissimilaridade Normalizada [0, 1] dos perfis encontrados, em função do tempo.
def spike_distance_t(spike_times):
    S_original = np.zeros(input_params['lif_simulation_time'])
    for t in range(input_params['lif_simulation_time']):
        S_original[t] = So(t, spike_times)
    #print('Spike Desejado: ', spike_times[0])
    #print('Spike Modelo: ', spike_times[1])
    #print('Dissimilaridade: ', S_original)    
    #S_original = (S_original-min(S_original))/(max(S_original)-min(S_original))
    return S_original

###########################################################################################
# Sinal de Erro baseado na Dissimilaridade total.
def dissimilarity_error(model_ST, desired_ST):
    d_profile = spike_distance_t([desired_ST, model_ST])
    return d_profile

###########################################################################################
# Dissimilaridade Total Media: Como estamos em tempo discreto, foi utilizado o metodo dos trapezios. 
# Em tempo contínuo seria a integral dos perfis de dissimilaridade.
def spike_total_distance (d_profile, T):
    return sum(d_profile)/(2*T)
    

###########################################################################################
# Exibe um comparativo entre 2 Spikes e Dissimilaridade entre eles
def print_spikes_diss(spikes, diss, label):
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    fig.suptitle('Comparando Spikes para o label {}'.format(label), fontsize=16)
    ax1.set_ylabel("Desired Spikes")
    ax1.step(range(input_params['lif_simulation_time']), spikes[0], linewidth=2.0)

    ax2.set_ylabel("Output Spikes")
    ax2.step(range(input_params['lif_simulation_time']), spikes[1], linewidth=2.0)

    ax3.set_ylabel("Dissimilaridade entre Desired e Output")
    ax3.plot(range(input_params['lif_simulation_time']), diss, linewidth=2.0)

    plt.show()


###########################################################################################
# Calcula a affinidade entre dois spikes, de acordo com o critério de Dissimilaridade
def spike_affinity(model_times=None, desired_times=None, model_spikes=None, desired_spikes=None, print_debug_label=-1):
    diss_vec = np.zeros(input_params['num_classes'])
    for label in range(input_params['num_classes']):
        times = np.asarray([desired_times[label],model_times[label]])
        diss_profile = spike_distance_t([desired_times[label],model_times[label]])
        if(print_debug_label > -1):
            spikes = np.asarray([desired_spikes[print_debug_label],model_spikes[print_debug_label]])
            print_spikes_diss(spikes, diss_profile, print_debug_label)
            print_debug_label = -1
        diss_vec[label] = spike_total_distance(diss_profile, input_params['lif_simulation_time'])
    #print(diss_vec)
    return np.argmin(diss_vec)


###########################################################################################
# Realiza o procedimento de predição de uma amostra de teste
def predict_label(test_samples=None, target_label=None, target_sample=None, print_spikes_diss=-1):
    output_spikes, _, output_times, _, _ = map(test_samples[target_label][target_sample], 
                                                        input_params['encoding_weights'],
                                                        input_params['mlp_weights'])
    
    label_affinity = spike_affinity(output_times, 
                                    input_params['desired_times'],
                                    output_spikes, 
                                    input_params['desired_spikes'], 
                                    print_debug_label = print_spikes_diss)

    return label_affinity

###########################################################################################
# Calcula a acurácia para cada classe.
def eval_accuracy(test_samples):
    acc = np.zeros(input_params['num_classes'])
    for label in range(input_params['num_classes']):
        num_acertos = 0
        for sample in range(input_params['num_test_samples']):
          #print("Sample {}, Label {}".format(sample, label))
          predicted_label = predict_label(test_samples, label, sample)
          #print('Rótulo Esperado: {} // Rótulo Predito: {}'.format(label, predicted_label))
          if(label == predicted_label): 
              num_acertos += 1
        acc[label] = num_acertos/input_params['num_test_samples']
        print("Acurácia para o rótulo {}: {}".format(label, round(100*(acc[label]),2)))
    return acc

###########################################################################################
# Roda um experimento com uma sequência de épocas para comparativo.
def run_epoch_experiment(epochs_list):
    epoch_cols = []
    acc_list = []
    
    for epoch in range(len(epochs_list)):
        #print('Usando {} épocas'.format(epochs_list[epoch]))
        input_params['num_epochs'] = epochs_list[epoch]
        epoch_cols.append('Época_'+str(epochs_list[epoch]))
        fit(samples)
        acc = eval_accuracy()
        acc_list.append(acc)

    return np.asarray(acc_list)