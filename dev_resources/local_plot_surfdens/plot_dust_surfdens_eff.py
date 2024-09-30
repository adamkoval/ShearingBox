#!/home/akoval/miniconda3/envs/python3.11_conda_env/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage
from argparse import ArgumentParser

# Parser
parser = ArgumentParser(description="Read and plot shearing box dump.")
parser.add_argument("-f", "--file", help="Dump file which to plot.")
parser.add_argument("-r", "--ranked", help="File containing list of ranked particles.")
args = parser.parse_args()
file = args.file
ranked = args.ranked
l = file.split('.')[-2][-1]

# Prepare output dir
if not os.path.exists("output"):
    print("Creating output directory.", flush=True)
    os.makedirs("output")

# Read in data
print("Reading data & converting to np array.", flush=True)
data = pd.read_csv(file, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None)

# Read in ranked list
print("Reading ranked list.", flush=True)
ranked = pd.read_csv(ranked, sep='\s+')

# Prepare arrays
surfdens = np.zeros((500,500))
surfdens_norm = np.zeros((500,500))

# (???)
mswarm = 1.44e-5

# Calculate j and k using vectorized operations
print("Adjusting x-y units.", flush=True)
data['j'] = ((data['x'] + 60.0) / (120.0 / 500.0)).astype(int) - 1
data['k'] = ((data['y'] + 60.0) / (120.0 / 500.0)).astype(int) - 1

# Apply the condition
print("Updating surface density.", flush=True)
condition = (data['size'] > 1.0e-15) & (data['size'] < 1.0e-11)

# Update surfdens matrix and num counter
num = 0
for j, k in data[condition][['j', 'k']].values:
    surfdens[j, k] += mswarm
    num += 1

surfdens = surfdens/(120.0/500.0)/(120.0/500.0)

avg_sdens = num*mswarm/120.0/120.0

surfdens_norm = surfdens/avg_sdens

image = []
for i in range(0,500):
    for j in range(0,500):
        if (surfdens_norm[i,j] < 0.0001):
            surfdens_norm[i,j] = 0.0001
        image.append(surfdens_norm[i,j])

image_log = np.log10(image)

hist_y, hist_x =  np.histogram(image, bins = 50, range=(0.01,100.0))

plt.loglog(hist_x[1:50], hist_y[0:49])
plt.xlabel('Relative density')
plt.ylabel('N')

plt.savefig(f'output/rel_dens_{l}.png')
plt.close()

image2D = np.reshape(np.ravel(image),[500,500])

image2D_log = np.log10(image2D)

rotated_img = ndimage.rotate(image2D_log,0.0)

plt.imshow(rotated_img,extent=[-60.0,60.0,-60.0,60.0],vmin=-1.0,vmax=2.0)
cbar = plt.colorbar(orientation='vertical')
cbar.set_label('log $(\Sigma_d/<\Sigma_d>)$',rotation=90)

plt.savefig(f'output/surfdens_{l}.png')
plt.close()

print(np.max(surfdens), np.max(surfdens_norm))