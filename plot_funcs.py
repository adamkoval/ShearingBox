import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage
from argparse import ArgumentParser

class PlotDustSurfdensEff:
    """
    Class to control the plotting of all the Accepts either a path to psliecout file or dct containing the data.
    In:
        > path_psliceout - (str) path to psliceout file
        > data - (dct) dictionary containing the data
        > path_ranked - (str) path to ranked file directory
    """
    def __init__(self, file, ranked):
        self.file = file
        self.ranked = ranked
        self.l = file.split('.')[-2][-1]
        self.surfdens = np.zeros((500,500))
        self.surfdens_norm = np.zeros((500,500))
        self.mswarm = 1.44e-5

    def read_data(self):
        data = pd.read_csv(self.file, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None)
        return data

    def read_ranked(self):
        ranked = pd.read_csv(self.ranked, sep='\s+')
        return ranked

    def adjust_units(self, data):
        data['j'] = ((data['x'] + 60.0) / (120.0 / 500.0)).astype(int) - 1
        data['k'] = ((data['y'] + 60.0) / (120.0 / 500.0)).astype(int) - 1
        return data

    def update_surfdens(self, data):
        condition = (data['size'] > 1.0e-15) & (data['size'] < 1.0e-11)
        num = 0
        for j, k in data[condition][['j', 'k']].values:
            self.surfdens[j, k] += self.mswarm
            num += 1
        self.surfdens = self.surfdens/(120.0/500.0)/(120.0/500.0)
        avg_sdens = num*self.mswarm/120.0/120.0
        self.surfdens_norm = self.surfdens/avg_sdens
        return self.surfdens, self.surfdens_norm

    def plot_surfdens(self):
        plt.imshow(self.surfdens_norm, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()