# This Python file uses the following encoding: utf-8
import numpy as np
class Izhikevich():
    def __init__(self, neuron_type=None):
        self.neuron_type = neuron_type
        self.setNeuronBaseProperties(self.neuron_type)
        self.v_peak = 30
    
    # Regular Spiking - RS
    # a = 0.02; b = 0.2; c = -65; d = 8; v0 = -60; C = 1; K = 0.7; vt = -40; v_peak = 30
    def regularSpikingType(self, u0=None):
        self.a = 0.02
        self.b = 0.2
        self.c = -65
        self.d = 8
        self.v = -60
        self.C = 1
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))
        
    # Intrinsically Bursting - IB
    # a = 0.02; b = 0.2; c = -55; d = 4; v0 = -60; C = 1; K = 0.7; vt = -40; v_peak = 30
    def intrinsicallyBurstingType(self, u0=None):
        self.a = 0.02
        self.b = 0.2
        self.c = -55
        self.d = 4
        self.v = -60
        self.C = 1
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))
    
    # Chattering - CH
    # a = 0.02; b = 0.2; c = -50; d = 2; v0 = -60; C = 1; K = 0.7; vt = -40; v_peak = 30
    def chatteringType(self, u0=None):
        self.a = 0.02
        self.b = 0.2
        self.c = -50
        self.d = 2
        self.v = -60
        self.C = 1
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))  
             
    # Fast Spiking - FS
    # a = 0.1; b = -2; c = -65; d = 1000; v0 = -60; C = 100; K = 0.7; vt = -40; v_peak = 30    
    def fastSpikingType(self, u0=None):
        self.a = 0.1
        self.b = -2
        self.c = -65
        self.d = 1000
        self.v = -60
        self.C = 100
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))
        
    # Thalamo-Cortical - TC
    # a = 0.1; b = 0.25; c = -65; d = 0.05; v0 = -60; C = 1; K = 0.7; vt = -40; v_peak = 30
    def thalamoCorticalType(self, u0=None):
        self.a = 0.1
        self.b = 0.25
        self.c = -65
        self.d = 0.05
        self.v = -60
        self.C = 1
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))
    
    # Resonator - RZ
    # a = 0.1; b = 0.25; c = -65; d = 2; v0 = -60; C = 1; K = 0.7; vt = -40; v_peak = 30
    def resonatorType(self, u0=None):
        self.a = 0.1
        self.b = 0.25
        self.c = -65
        self.d = 2
        self.v = -60
        self.C = 1
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))
        
    # Low-threshold spiking - LTS
    #a = 0.02; b = 0.25; c = -65; d = 2; v0 = -60; C = 1; K = 0.7; vt = -40; v_peak = 30   
    def lowThresholdType(self, u0=None):
        self.a = 0.02
        self.b = 0.25
        self.c = -65
        self.d = 2 
        self.v = -60
        self.C = 1
        self.vt = -40
        self.K = 0.7
        self.u = u0 if u0 is not None else self.b*(-60)
        self.vr = -60
        self.du = lambda a, b, v, u, vr: (self.a*(self.b*(self.v - self.vr) - self.u))
    
    def setNeuronBaseProperties(self, type=None):
        if (type == 'LT'): self.lowThresholdType()
        elif (type == 'IB'): self.intrinsicallyBurstingType()
        elif (type == 'CH'): self.chatteringType()
        elif (type == 'FS'): self.fastSpikingType()
        elif (type == 'TC'): self.thalamoCorticalType()
        elif (type == 'RZ'): self.resonatorType()
        else: self.regularSpikingType()
            
    def evaluatePotential(self, current, dt):
        v_t = (dt/self.C) * ((self.K*(self.v - self.vr)*(self.v -self.vt)) - self.u + current) 
        u_t = dt * self.du(self.a, self.b, self.v, self.u, self.vr)
        return v_t, u_t
    
    def restAfterSpike(self):
        self.v = self.c
        self.u += self.d

    def spikeResponse(self, current, dt, spike_train, neuron_pot, spike_time):
        timeline = len(current)
        spike_count = 0

        for t in range(timeline):
            v_t, u_t = self.evaluatePotential(current[t], dt)
            self.v += v_t
            self.u += u_t
            neuron_pot[t] = self.v
            if (self.v >= self.v_peak):
                spike_count += 1
                spike_train[t] = 1
                spike_time[t] = t
                self.restAfterSpike()
        return spike_count

class Synapse():
    def __init__(self, patches_ds):
        self.input_data = patches_ds
    
    def setSynapticWeightMatrix(self):
        self.synW = np.random.uniform(0, 1, (self.input_data.shape[0], 
                                             self.input_data.shape[1], self.input_data.shape[2]))
    
    def setPostSynapticSignal(self):
        pass
        