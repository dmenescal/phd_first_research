# This Python file uses the following encoding: utf-8
# Image preprocessing: Patches extraction
import numpy as np

class ImagePreProcessing():
    def __init__(self, dataset_patterns, dataset_labels, patch_size, patch_stride):
        self.patterns = dataset_patterns/255
        self.labels = dataset_labels
        self.patch_size = patch_size
        self.n = self.patterns.shape[1]
        self.num_patches = (self.n-self.patch_size)*(self.n-self.patch_size)
        self.max_patterns = 5000
        self.num_labels = 10
        self.patch_stride = patch_stride

    def getPatternsByLabel(self, label):
        return self.patterns[np.where(self.labels == label)]
    
    def setMaxPatterns(self, val):
        self.max_patterns = val 

    def setMaxLabels(self, val):
        self.num_labels = val
    
    # Get one image pattern for each label. Just for sampling.
    def getImageSetSample(self):
        ps = []
        for i in range(self.num_labels):
            p = self.getPatternsByLabel(i)
            end = np.random.randint(0, self.max_patterns-1)
            ps.append(p[end])
        return np.asarray(ps)
    
    # V1: Sliding Patches, with stride = 1
    def extractPatchesFromImage(self, image):
        patches = []
        for i in range(self.n):
            for j in range(self.n):
                pivot = (i,j)
                if(pivot[0]+self.patch_size >= self.n or pivot[1]+self.patch_size >= self.n): continue
                patch = image[pivot[0]:pivot[0]+self.patch_size, pivot[1]:pivot[1]+self.patch_size]
                patches.append(patch)
        return np.asarray(patches)
    
    def extractPatchesFromImage_V2(self, image, stride):
        patches = []
        ci = 0
        while(ci < self.n):
            cj = 0
            while(cj < self.n):
                pivot = (ci, cj)
                if(pivot[0]+self.patch_size >= self.n or pivot[1]+self.patch_size >= self.n):
                    cj += stride
                else:
                    patch = image[pivot[0]:pivot[0]+self.patch_size, pivot[1]:pivot[1]+self.patch_size]
                    patches.append(patch)
                    cj += stride
            ci += stride
        return np.asarray(patches)

    
    def buildDataset(self):
        pats_set = []
        for i in range(self.num_labels):
            ps = self.getPatternsByLabel(i)
            ps = ps[0:self.max_patterns] # Limits for an equal number of patterns for each label
            pats_set.append(ps)
            #printImg(ps[0], i+1)
        self.patlab_ds = np.asarray(pats_set)

    def buildPatchesSet(self):
        labels = self.patlab_ds.shape[0]
        pats = self.patlab_ds.shape[1]
        label_patches = []
        for label in range(labels):
            pattern_patches = []
            for pat in range(pats):
                #patches = self.extractPatchesFromImage(self.patlab_ds[label][pat])
                patches = self.extractPatchesFromImage_V2(self.patlab_ds[label][pat], self.patch_stride)
                pattern_patches.append(patches)
            label_patches.append(pattern_patches)
        self.patches_ds = np.asarray(label_patches)
        return self.patches_ds

    def summary(self):
        print("Dimensão (n x n) da imagem de entrada: n = {}".format(self.n))
        print("Dimensão (p x p) de cada patch(subset) da imagem: p = {}".format(self.patch_size))
        print("Deslocamento da janela de extração de patches: {}".format(self.patch_stride))
        print("Quantidade de rótulos utilizados: {}".format(self.num_labels))
        print("Quantidade de padrões utilizados: {}".format(self.max_patterns))
        print("Quantidade de patches extraídos por imagem: {}".format(self.patches_ds.shape[2]))
        print("Dimensão do conjunto de imagens, por rótulo: {}".format(self.patlab_ds.shape))
        print("Dimensões do conjunto de patches (Entrada para SNN): {}".format(self.patches_ds.shape))

    def run(self):
        self.buildDataset()
        return self.buildPatchesSet()