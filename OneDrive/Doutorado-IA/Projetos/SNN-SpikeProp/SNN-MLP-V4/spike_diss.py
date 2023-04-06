import numpy as np


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
    return S_original

###########################################################################################
# Dissimilaridade Total Media: Como estamos em tempo discreto, foi utilizado o metodo dos trapezios. 
# Em tempo contínuo seria a integral dos perfis de dissimilaridade.
def spike_total_distance (d_profile, T):
    return sum(d_profile)/(2*T)

###########################################################################################
# Sinal de Erro baseado na Dissimilaridade total.
def dissimilarity_error(model_ST, desired_ST):
    d_profile = spike_distance_t([desired_ST, model_ST])
    return d_profile
    

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
    return np.argmin(diss_vec)