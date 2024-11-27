import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage
from argparse import ArgumentParser

import func_gen as gf

class PlotDustSurfdensEff:
    def __init__(self, fin, data, COMs, R_Hs, n_neighbors, res=500):
        """
        Class to control the plotting of all the 
        Accepts either a path to psliecout file or dct containing the data.
        In:
            > data - (dct) dictionary containing the data
            > COMs - (dct) dictionary containing the COMs
            > R_Hs - (dct) dictionary containing the Hill radii of COMs
            > n_neighbor - (int) number of neighbors used in kD search
        """
        # Report function name
        print(gf.report_function_name(), flush=True)
        print(f"\tPlotting dust surface density for {fin}", flush=True)

        # Assign variables
        self.data = data
        self.COMs = COMs
        self.R_Hs = R_Hs
        self.N = n_neighbors
        self.fin = fin
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
        self.log_image2D = np.log10(self.image2D)

    def add_R_Hs(self):
        COMs_x = np.array(self.COMs['x'])
        COMs_y = np.array(self.COMs['y'])
        for i in range(len(COMs_x)):
            circle_H = plt.Circle((COMs_y[i], -COMs_x[i]), self.R_Hs[i], color='r', fill=False)
            self.ax.add_patch(circle_H)

    def plot_surfdens(self, save=False, show=True):
        self.fig, self.ax = plt.subplots()
        self.adjust_units(self.data)
        self.update_surfdens(self.data)
        self.populate_image()
        self.add_R_Hs()
        surfdens_plot = self.ax.imshow(self.log_image2D,
                   extent=[-60.0,60.0,-60.0,60.0],
                   vmin=-1.0,vmax=2.0)
        cbar = plt.colorbar(surfdens_plot, orientation='vertical', ax=self.ax)
        cbar.set_label('log $(\Sigma_d/<\Sigma_d>)$', rotation=90)
        self.ax.set_xlabel('y')
        self.ax.set_ylabel('x')
        self.ax.set_title(self.fin)
        if save:
            plt.savefig(f"output/surfdensplots/surfdens_{self.N}nigh_{self.fin}.pdf", format='pdf')
        if show:
            plt.show()