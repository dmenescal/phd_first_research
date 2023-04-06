# This Python file uses the following encoding: utf-8
# Construção do Modelo de Neurônio do tipo Izhikevich, baseado no seguinte material:
# Training Spiking Neurons by Means of Particle Swarm Optimization
    
class IzhNeuron:
    def __init__(self, label, a, b, c, d, v0, C, K, vt, v_peak, u0=None):
        self.label = label
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = v0
        self.u = u0 if u0 is not None else b*v0
        self.C = C
        self.vt = vt
        self.K = K
        self.vr = v0
        self.v_peak = v_peak
    