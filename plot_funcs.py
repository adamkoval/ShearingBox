import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage
from argparse import ArgumentParser

import gen_funcs as gf

class PlotDustSurfdensEff:
    """
    Class to control the plotting of all the Accepts either a path to psliecout file or dct containing the data.
    In:
        > data - (dct) dictionary containing the data
        > COMs - (dct) dictionary containing the COMs
        > R_Hs - (dct) dictionary containing the Hill radii of COMs
    """
    def __init__(self, data, COMs, R_Hs, res=500):
        self.data = data
        self.COMs = COMs
        self.R_Hs = R_Hs
        # self.l = file.split('.')[-2][-1]
        self.res = res
        self.surfdens = np.zeros((self.res, self.res))
        self.surfdens_norm = np.zeros((self.res, self.res))

    def adjust_units(self, data):
        data_x = data['raw_coords'][:, 0]
        data_y = data['raw_coords'][:, 1]
        self.data['j'] = ((data_x + 60.0) / (120.0 / float(self.res))).astype(int) - 1
        self.data['k'] = ((data_y + 60.0) / (120.0 / float(self.res))).astype(int) - 1

    def update_surfdens(self, data):
        data_size = data['raw_sizes']
        condition = (data_size > 1.0e-15) & (data_size < 1.0e-11)
        num = 0
        stacked = np.vstack((self.data['j'][condition], self.data['k'][condition]))
        for j, k in stacked.T:
            self.surfdens[j, k] += gf.m_swarm
            num += 1
        self.surfdens = self.surfdens/(120.0/float(self.res))/(120.0/float(self.res))
        avg_sdens = num*gf.m_swarm/120.0/120.0
        self.surfdens_norm = self.surfdens/avg_sdens

    def populate_image(self):
        image1D = []
        for i in range(self.res):
            for j in range(self.res):
                if (self.surfdens_norm[i,j] < 0.0001):
                    self.surfdens_norm[i,j] = 0.0001
                image1D.append(self.surfdens_norm[i,j])
        self.image2D = np.array(image1D).reshape(self.res, self.res)

    def add_R_Hs(self):
        COMs_x = np.array(self.COMs['x'])
        COMs_y = np.array(self.COMs['y'])
        for i in range(len(COMs_x)):
            circle_H = plt.Circle((COMs_y[i], -COMs_x[i]), self.R_Hs[i], color='r', fill=False)
            plt.gca().add_artist(circle_H)

    def plot_surfdens(self):
        self.adjust_units(self.data)
        self.update_surfdens(self.data)
        self.populate_image()
        self.add_R_Hs()
        plt.imshow(np.log10(self.image2D),
                   extent=[-60.0,60.0,-60.0,60.0],
                   vmin=-1.0,vmax=2.0)
        cbar = plt.colorbar(orientation='vertical')
        cbar.set_label('log $(\Sigma_d/<\Sigma_d>)$',rotation=90)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.show()