# This Python file uses the following encoding: utf-8
from snn_sample import IzhSample
from snn_setup import SNNSetup
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import trange

class LabParam():
    def __init__(self,
                 param_desc,
                 num_patterns,
                 num_labels,
                 target_label,
                 patch_size,
                 synaptic_weights,
                 current_gama,
                 simulation_time,
                 simulation_delta,
                 neuron_model_params):
        
        self.np = num_patterns
        self.nl = num_labels
        self.tl = target_label
        self.ps = patch_size
        self.sw = synaptic_weights
        self.cg = current_gama
        self.st = simulation_time
        self.sd = simulation_delta
        self.nmp = neuron_model_params
        self.desc = param_desc
        
class LabExp():
    def __init__(self, title):
        self.title = title
        self.labParams = []
    
    def registerParameter(self, param):
        self.labParams.append(param)
        
    def run_experiment(self, plot_results=False):
        stats_results = []
        for i in trange(len(self.labParams), desc="Running experiments..."):
            param = self.labParams[i]
            neuron_sample = IzhSample(target_label     = param.tl,
                                      simulation_time  = param.st,
                                      simulation_delta = param.sd,
                                      neuron_params    = param.nmp)
            
            snn_setup     = SNNSetup(num_labels   = param.nl,
                                     num_patterns = param.np,
                                     current_gama = param.cg,
                                     weights      = param.sw,
                                     patch_size   = param.ps,
                                     target_label = param.tl)
            d      = snn_setup.build_set_by_labels_and_patterns()
            I_pats = snn_setup.calculate_input_current(d)
            spikes = neuron_sample.generate_spike_samples(I_pats)
            spike_stats = neuron_sample.generate_samples(spikes, plot_results, param.desc)
            stats_results.append(spike_stats)
        return stats_results
    
class LabResultAnalisys():
    def __init__(self, exp_results, title):
        self.title = title
        self.exp_results = exp_results
    
    def __prepare_plotting_set(self, result):
        nlabels = result.shape[0]
        npatterns = result.shape[1]
        
        df = []
        for i in range(nlabels):
            for j in range(npatterns):
                df.append([result[i][j], i])
        
        return pd.DataFrame(data=df, columns=["spike_count","spike_label"])
        
    def __plot_spike_histogram(self, result):
        nPat = len(result[0])
        plt.figure(figsize=(16,8))
        plt.title(self.title + ": " + str(nPat) + " Patterns per Label")
        plot_data = self.__prepare_plotting_set(result)    
        sns.kdeplot(
           data=plot_data, x="spike_count", hue="spike_label",
           fill=True, common_norm=False, palette="tab10",
           alpha=1.0, linewidth=0,
        )
    
    def plot_histogram_analisys(self):
        for i in trange(len(self.exp_results), desc="Generating Histogram..."):
            self.__plot_spike_histogram(self.exp_results[i])
        
        


        
        
            
