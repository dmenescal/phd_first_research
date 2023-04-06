import snn_params

'''
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
    desired_spike_bin = np.zeros((snn_params.input_params['num_classes'], snn_params.input_params['lif_simulation_time']))
    num_classes = snn_params.input_params['num_classes']
    
    for i in range(num_classes):
        spike_time = random.sample(range(spike_range[i][0], spike_range[i][1]), int(spike_count[i]))
        desired_spike_bin[i][spike_time] = 1
        desired_spike_train.append(np.sort(spike_time))
    
    if (class_idx >= 0):
        return  desired_spike_bin[class_idx]
    
    return desired_spike_bin, np.asarray(desired_spike_train)

###########################################################################################
## Main function to return desired spikes
def get_desired_spikes():
    desired_range = spike_interval_generator(simulation_time=snn_params.input_params['lif_simulation_time'])
    snn_params.set_param('desired_range', desired_range)
    desired_count = spike_count_generator(simulation_time=snn_params.input_params['lif_simulation_time'])
    snn_params.set_param('desired_count', desired_count)
    dsp = spike_times_generator(desired_range, desired_count, -1)
    return dsp

'''

