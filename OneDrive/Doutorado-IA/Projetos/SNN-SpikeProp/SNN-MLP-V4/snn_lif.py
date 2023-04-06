import math
import snn_params

###########################################################################################
## LIF Neuron Potential Dynamics
def lif_activation(R, I, V_rest, V0, tao, t):
    return (V_rest+(R*I))+(V_rest-(R*I)-V0)*math.exp((-1*t)/(tao))

###########################################################################################
## LIF Activation Derivative
def lif_activation_derivative(R, I, V_rest, V0, tao, t):
    return ((V_rest+(R*I)-V0)*math.exp((-1*t)/(tao)))/tao  

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
## Encoding Layer Spikes Generation - V1
def encoding_layer_spikes_generation(input_current, neuron_resistance, time_constant, 
                                     rest_potential, start_potential, threshold, simulation_time):
    st = []
    st_times = []
    st_count = []

    spike_train_param = 'encoding_spikes'
    spike_time_param  = 'encoding_spike_times'
    spike_count_param = 'encoding_spike_count'
    
    for neuron in range(snn_params.input_params['patches_set']):
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

    snn_params.set_param(spike_train_param, st)
    snn_params.set_param(spike_time_param, st_times)
    snn_params.set_param(spike_count_param, st_count)

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
    
    snn_params.set_param(spike_train_param, spike_bin)
    snn_params.set_param(spike_time_param, spike_times)
    snn_params.set_param(spike_count_param, spike_count)

    
    return spike_bin, spike_times, spike_count