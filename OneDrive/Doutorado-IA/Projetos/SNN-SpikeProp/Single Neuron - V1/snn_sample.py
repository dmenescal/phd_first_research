# This Python file uses the following encoding: utf-8
from snn_model import IzhNeuron
from snn_simulate import IzhSim

import matplotlib.pyplot as plt
import numpy as np

class IzhSample():
    def __init__(self, 
                 title            = "Neuron Spike Generation, Sampling and Viz",
                 neuron_params    = [0.03, -2, -50, 100, -60, 100, 0.7, -40, 35], 
                 target_label     = -1, 
                 simulation_time  = 100, 
                 simulation_delta = 0.1): 
        self.title = title
        self.label = target_label
        self.T     = simulation_time
        self.dT    = simulation_delta
        self.neuron_params = neuron_params
    
    def generate_spike_samples(self, input_current):
        sims = []
        
        self.nlabels = input_current.shape[0]
        self.npatterns = input_current.shape[1]
        
        if(self.label > 0):
            for pattern in range(input_current.shape[1]):
                title = "Spiking Neuron, image sampling: Label= " + str(self.label) + ", Pattern=" + str(pattern)
                n = IzhNeuron(title, a = self.neuron_params[0], 
                                     b = self.neuron_params[1], 
                                     c = self.neuron_params[2], 
                                     d = self.neuron_params[3], 
                                     v0 = self.neuron_params[4], 
                                     C = self.neuron_params[5], 
                                     K = self.neuron_params[6],
                                     vt = self.neuron_params[7],
                                     v_peak = self.neuron_params[8]) # Create here a neuron network
                s = IzhSim(n, T = self.T, dt=self.dT)
                for i, t in enumerate(s.t):
                    s.stim[i] = input_current[0][pattern]
                sims.append(s) 
        else:
            for label in range(input_current.shape[0]):
                for pattern in range(input_current.shape[1]):
                    title = "Spiking Neuron, image sampling: Label= " + str(label) + ", Pattern=" + str(pattern)
                    n = IzhNeuron(title, a = self.neuron_params[0], 
                                         b = self.neuron_params[1], 
                                         c = self.neuron_params[2], 
                                         d = self.neuron_params[3], 
                                         v0 = self.neuron_params[4], 
                                         C = self.neuron_params[5], 
                                         K = self.neuron_params[6],
                                         vt = self.neuron_params[7],
                                         v_peak = self.neuron_params[8]) # Create here a neuron network
                    s = IzhSim(n, T = self.T, dt=self.dT)
                    for i, t in enumerate(s.t):
                        s.stim[i] = input_current[label][pattern]
                    sims.append(s)
        return sims

    def generate_samples(self, spikes, plot, title):
        spike_stats = []
        if(plot == True):
            plt.figure(figsize=(16,8))
        for i,s in enumerate(spikes):
            res, spk_count = s.integrate()
            spike_stats.append(spk_count)
            if(plot == True):
                ax = plt.subplot(5,2,i+1)
                val = (s.stim/max(s.stim))*10
                ax.plot(s.t, res[0], s.t, val)
                ax.set_xlim([0, s.t[-1]])
                ax.set_ylim([-100,self.neuron_params[8]])
                ax.set_title(title)
                #ax.set_title(s.neuron.label + ", Spike count = "+ str(spk_count))
        if(plot == True):
            plt.show()
        
        ss = np.asarray(spike_stats)
        return ss.reshape(self.nlabels, self.npatterns)