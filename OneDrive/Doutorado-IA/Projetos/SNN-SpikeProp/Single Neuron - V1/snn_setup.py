# This Python file uses the following encoding: utf-8
import numpy as np
from keras.datasets import mnist


class SNNSetup():
    def __init__(self, 
                 num_labels   = 10, 
                 num_patterns = 10, 
                 current_gama = 1, 
                 weights      = np.random.uniform(0, 1, (5, 5)),
                 patch_size   = 5, 
                 target_label = -1):
        self.nLabels   = num_labels
        self.nPatterns = num_patterns
        self.gama      = current_gama
        self.W         = weights
        self.p         = patch_size
        self.tLabel    = target_label
        (pats, labs)   = self.__get_mnist_dataset()
        self.patterns  = pats
        self.labels    = labs
        
    def __get_unique_label_set(self, target):
        pat = self.patterns[np.where(self.labels == target)]
        return pat[0:self.nPatterns]
    
    def __extract_patches_from_stimuli(self, data):
        n = data.shape[0]
        patches = []
        for i in range(n):
            for j in range(n):
                pivot = (i,j)
                if(pivot[0]+self.p >= n or pivot[1]+self.p >= n): continue
                patch = data[pivot[0]:pivot[0]+self.p, pivot[1]:pivot[1]+self.p]
                patches.append(patch)
        return np.asarray(patches)

    def __set_membrane_current(self, patches):
        n = self.W.shape[0]
        acc = 0
        for patch in patches:
            for i in range(n):
                for j in range(n):
                    acc = acc + (self.W[i][j]*patch[i][j])
        return acc
   
    def __get_mnist_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train/255, y_train)

    def build_set_by_labels_and_patterns(self):
        D = []
        if(self.tLabel > 0):
            D.append(self.__get_unique_label_set(self.tLabel))
        else:
            for i in range(self.nLabels):
                D.append(self.__get_unique_label_set(i))
        return D
    
    def calculate_input_current(self, dataset):
        I_pats = np.empty((self.nLabels, self.nPatterns))
        for i in range(len(dataset)):
            for j in range(self.nPatterns):
                patches = self.__extract_patches_from_stimuli(dataset[i][j])
                I_pats[i][j] = self.gama*self.__set_membrane_current(patches)
        return I_pats
    