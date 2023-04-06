# This Python file uses the following encoding: utf-8
import matplotlib.pyplot as plt

class SpikeViz():
    def __init__(self, original_image):
        self.original_image = original_image
        self.setupImageProps()
    
    def setupImageProps(self):
        self.figsize = (30,40)
        self.nrows = 10
        self.ncols = 2
    
    def printSpikeViz(self, spike_obj, x_label, y_label, by_neuron, neuron_idx=None):
        fig, ax = plt.subplots(self.nrows,self.ncols, figsize=self.figsize)
        for i in range(10):
            ax[i,0].imshow(self.original_image[i][0], cmap='gray')
            ax[i,0].set_title("Dígito")
            ax[i,0].axis('off')
            if (by_neuron == False):
                ax[i,1].plot(spike_obj[i][0])
            else:
                if (spike_obj.shape[2] == 1):
                    ax[i,1].bar(0, spike_obj[i][0][neuron_idx])
                ax[i,1].plot(spike_obj[i][0][neuron_idx])
            ax[i,1].set(xlabel=x_label, ylabel=y_label)
