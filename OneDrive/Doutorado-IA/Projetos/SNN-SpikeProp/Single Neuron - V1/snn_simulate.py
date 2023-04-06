# This Python file uses the following encoding: utf-8
import numpy as np

# Simulação do Neurônio dentro de um intervalo de tempo T.
# Define também um intervalo de tempo dT para analisar os valores de potencial da membrana.
# Aqui, a entrada/estímulo é definida como padrão um array zerado.
class IzhSim:
    def __init__(self, n, T, dt=0.25):
        self.neuron = n
        self.dt = dt
        self.t = t = np.arange(0, T+dt, dt)
        self.stim = np.zeros(len(t)) # Este estímulo é apenas um array 1D. Vou precisar adicionar um 2D para receber imagens
        self.x = 70
        self.y = 1680
        self.du = lambda a, b, v, u, vr: (a*(b*(v - vr) - u))

    def integrate(self, n=None):
        spikes = 0
        if n is None: n = self.neuron
        trace = np.zeros((2,len(self.t)))
        for i, j in enumerate(self.stim): #Alterar esse loop para tratar a imagem como entrada
            n.v += (self.dt/n.C) * ((n.K*(n.v - n.vr)*(n.v - n.vt)) - n.u + self.stim[i]) 
            n.u += self.dt * self.du(n.a,n.b,n.v,n.u,n.vr)
            if n.v >= n.v_peak:
                spikes += 1
                trace[0,i] = n.v_peak
                n.v = n.c
                n.u += n.d
            else:
                trace[0,i] = n.v
                trace[1,i] = n.u
        return trace, spikes