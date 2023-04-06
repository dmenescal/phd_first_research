# This Python file uses the following encoding: utf-8
import numpy as np
from spiking_neuron import Izhikevich

class SpikeRepresentationLayer():
    def __init__(self, patches, weights, neurons_type):
        self.bio_params = {}
        self.weights = weights
        self.patches = patches
        self.num_labels = patches.shape[0]
        self.num_patterns = patches.shape[1]
        self.num_neurons = patches.shape[2]
        self.setTimeParams()
        self.setMembraneCurrent()
        self.setInputStimuli()
        self.neurons_types = neurons_type
        
    def setTimeParams(self, T=100, dt=0.25):
        self.dt = dt
        self.t = t = np.arange(0, T+dt, dt)
        self.stim = np.zeros((self.num_labels,self.num_patterns, self.num_neurons,len(t) ))
    
    def resetBioParams(self):
        self.bio_params = Izhikevich(neuron_type=self.neurons_types)
        self.du = lambda a, b, v, u, vr: (self.bio_params.a*(self.bio_params.b*(self.bio_params.v - self.bio_params.vr) - self.bio_params.u))
        
    def setMembraneCurrent(self, gama=100):
        self.current = np.zeros((self.num_labels, self.num_patterns, self.num_neurons))
        for label in range(self.patches.shape[0]):
            for pattern in range(self.patches.shape[1]):
                acc = 0
                for neuron in range(self.patches.shape[2]):
                    for i in range(self.patches.shape[3]):
                        for j in range(self.patches.shape[4]):
                            acc += self.patches[label][pattern][neuron][i][j]*self.weights[neuron][i][j]
                    self.current[label][pattern][neuron] = gama*acc
                    acc = 0
    
    def setInputStimuli(self):
        for label in range(self.stim.shape[0]):
            for pattern in range(self.stim.shape[1]):
                for neuron in range(self.stim.shape[2]):
                    for t in range(self.stim.shape[3]):
                        self.stim[label][pattern][neuron][t] = self.current[label][pattern][neuron]

    def evaluatePotential(self, current):
        self.bio_params.v += (self.dt/self.bio_params.C) * ((self.bio_params.K*(self.bio_params.v - self.bio_params.vr)*(self.bio_params.v -self.bio_params.vt)) - self.bio_params.u + current) 
        self.bio_params.u += self.dt * self.du(self.bio_params.a,self.bio_params.b,self.bio_params.v,self.bio_params.u,self.bio_params.vr)  

    def refreshPotentialAfterSpike(self):
        self.bio_params.v = self.bio_params.c
        self.bio_params.u += self.bio_params.d

    def registerSpike(self, label, pattern, neuron, spike_time):
        self.neuron_spikes[label][pattern][neuron] += 1
        self.net_spike_train[label][pattern][neuron][spike_time] = 1
    
    def generateSpikeTrain(self):
        self.neuron_spikes = np.zeros((self.num_labels, self.num_patterns, self.num_neurons))

        # Instantiate Spike train with -1. This means no spike at no time.
        self.net_spike_train = np.zeros((self.num_labels, self.num_patterns, self.num_neurons, self.stim.shape[3]))
        
        self.net_neuron_potential = np.zeros((self.num_labels, self.num_patterns, self.num_neurons, self.stim.shape[3]))

        for label in range(self.num_labels):
            for pattern in range(self.num_patterns):
                for neuron in range(self.stim.shape[2]): # For each neuron on layer
                    self.resetBioParams()
                    for t in range(self.stim.shape[3]): # For each step of time
                        self.evaluatePotential(self.stim[label][pattern][neuron][t])
                        self.net_neuron_potential[label][pattern][neuron][t] = self.bio_params.v
                        if (self.bio_params.v >= self.bio_params.v_peak):
                            self.registerSpike(label, pattern, neuron, t)
                            self.refreshPotentialAfterSpike()

        return self.net_spike_train, self.neuron_spikes