import random
import matplotlib.pyplot as plt
import numpy as np
import vispy.scene
from vispy.scene import visuals

code_L = 1.496e13 # Code Length unit, in cm
code_M = 1.989e33 # Code Mass unit, in g

def get_plotting_data(data1, xclump, yclump, zclump):
    """
    Gets plotting data like Ken did but stores it for working on further instead of plotting.
    """
    print("Working with second dataset & plotting scatter")
    units = code_M/code_L/code_L
    #for i in range(0,np.shape(data1[:,0])[0]):
    xdat = np.empty()
    ydat_Stam = np.empty()
    ydat_Lomb = np.empty()
    ydat_comb = np.empty()
    for i in range(0,2000000):
        xdist = xclump-data1[i,0]
        ydist = yclump-data1[i,1]
        zdist = zclump-data1[i,2]
        dist = np.sqrt(xdist**2.0+ydist**2.0+zdist**2.0)
        xdat = np.append(xdat, dist)

        # For now, keep it simple and just do a subset like Ken has..
        if (dist < 10.0):
            if (random.random() > 0.985):
                ydat_Stam = np.append(ydat_Stam, data1[i,4]*data1[i,3]*units)
                ydat_Lomb = np.append(ydat_Lomb, data1[i,5]*data1[i,3]*units)
                ydat_comb = np.append(ydat_comb, data1[i,6]*data1[i,3]*units)
    return xdat, ydat_Stam, ydat_Lomb, ydat_comb


def plot_Ken(data1, xclump, yclump, zclump, clump_rad, clump_coldens):
    """
    Snippet of Ken's code for plotting SPH density profile vs. approximations
    In:
        > data1 - (arr) SPH array from Phantom which includes scale height approximations (all data in code units)
        > xclump, yclump, zclump - (floats) centre point coords of clump
        > clump_rad - (arr) radial distances of each shell
        > clump_coldens - (arr) column density of each shell
    """
    # Analysing Phantom run with approximations
    print("Working with second dataset & plotting scatter")
    units = code_M/code_L/code_L
    #for i in range(0,np.shape(data1[:,0])[0]):
    for i in range(0, 2000000):
        xdist = xclump - data1[i,0]
        ydist = yclump - data1[i,1]
        zdist = zclump - data1[i,2]
        dist = np.sqrt(xdist**2.0 + ydist**2.0 + zdist**2.0)

        # Plot approximations
        if (dist < 10.0):
            if (random.random() > 0.985):
                plt.scatter(dist, data1[i,4]*data1[i,3]*units, s=20.0, marker='o', facecolor='none', edgecolor='green')
                plt.scatter(dist, data1[i,5]*data1[i,3]*units, s=20.0, marker='s', facecolor='none', edgecolor='red')
                plt.scatter(dist, data1[i,6]*data1[i,3]*units, s=20.0, marker='*', facecolor='none', edgecolor='blue')
                #plt.scatter(dist, data1[i,3]*units, s=20.0,marker='^',facecolor='none',edgecolor='black') # test
                #print(i,dist)

    # Plot single point for each approximation, to be used in legend
    plt.scatter(0.0, 0.0, s=20.0, marker='o', facecolor='none', edgecolor='green', label='Stamatellos')
    plt.scatter(0.0, 0.0, s=20.0, marker='s', facecolor='none', edgecolor='red', label='Lombardi')
    plt.scatter(0.0, 0.0, s=20.0, marker='*', facecolor='none', edgecolor='blue', label='Combined')

    # Plot 2000 particles from full SPH run
    #plt.plot(clump_rad,clump_rho,color='black',lw=2.0)
    plt.plot(clump_rad[0:2000], clump_coldens[0:2000], color='black', ls='dashed', lw=2.0, label='SPH')

    # Formatting
    plt.xlim(0.01, 10.0)
    plt.ylim(0.1, 100000)
    plt.yscale('log')
    plt.xlabel('Clump radius (AU)')
    #plt.ylabel('Density (g cm$^{-3}$)')
    plt.ylabel('Column density (g cm$^{-2}$)')
    plt.legend(loc='upper right')
    plt.savefig('Clump_column_density.png')
    plt.show()


def plot_denssurf(cyl_rad, cyl_denssurf):
    """
    Plots shell densities divided by shell surface areas
    In:
        > cyl_rad - (arr) radial distance of each shell
        > cyl_denssurf - (arr) shell densities divided by shell surface area
    """
    # Plotting full SPH run with current analysis
    print("Plotting denssurf from SPH")
    plt.figure()
    plt.plot(cyl_rad[0:2000], cyl_denssurf[0:2000], color='black', ls='-', lw=.5, label='SPH')
    #plt.xlim(0.01, 10.0)
    #plt.ylim(0.1, 100000)
    plt.yscale('log')
    plt.xlabel('Cylinder radius (AU)')
    plt.ylabel('Column density/surface area (gcm$^{-5}$)')
    plt.legend(loc='upper right')
    plt.savefig('cylinder_columndensityoverSA.png')
    plt.show()


def plot_masssurf(cyl_rad, cyl_masssurf):
    """
    Plots shell total shell masses divided by shell surface areas
    In:
        > cyl_rad - (arr) radial distance of each shell 
        > cyl_masssurf - (arr) shell mass divided by shell surface area
    """
    # Plotting full SPH run with current analysis
    print("Plotting masssurf from SPH")
    plt.figure()
    plt.plot(cyl_rad[0:2000], cyl_masssurf[0:2000], color='black', ls='-', lw=.5, label='SPH')
    #plt.xlim(0.01, 10.0)
    #plt.ylim(0.1, 100000)
    plt.yscale('log')
    plt.xlabel('Cylinder radius (AU)')
    plt.ylabel('Column mass/surface area (gcm$^{-2}$)')
    plt.legend(loc='upper right')
    plt.savefig('cylinder_columnmassoverSA.png')
    plt.show()


def plot_surfcoldens(cyl_rad, cyl_surfcoldens):
    """
    Plots shell total shell masses divided by shell surface areas
    In:
        > cyl_rad - (arr) radial distance of each shell 
        > cyl_surfcoldens - (arr) shell mass divided by shell top-down surface area (in gcm-2)
    """
    # Plotting full SPH run with current analysis
    print("Plotting surfcoldens from SPH")
    plt.figure()
    plt.plot(cyl_rad[0:2000], cyl_surfcoldens[0:2000], color='black', ls='-', lw=.5, label='SPH')
    #plt.xlim(0.01, 10.0)
    #plt.ylim(0.1, 100000)
    plt.yscale('log')
    plt.xlabel('Cylinder radius (AU)')
    plt.ylabel('Column mass/top-down surface area (gcm$^{-2}$)')
    plt.legend(loc='upper right')
    plt.savefig('cylinder_columnmassoverTDSA.png')
    plt.show()


def plot_Adam(data1, xclump, yclump, zclump, cyl_rad, cyl_surfcoldens):
    """
    Using the sippet of Ken's code for plotting SPH density profile vs. approximations
    In:
        > data1 - (arr) SPH array from Phantom which includes scale height approximations (all data in code units)
        > xclump, yclump, zclump - (floats) centre point coords of clump
        > cyl_rad - (arr) radial distances of each shell
        > cyl_surfcoldens - (arr) column mass of each shell divided by shell top-down surface area (in gcm-2)
    """
    # Analysing Phantom run with approximations
    print("Working with second dataset & plotting scatter")
    units = code_M/code_L/code_L
    #for i in range(0,np.shape(data1[:,0])[0]):
    for i in range(0, 2000000):
        xdist = xclump - data1[i,0]
        ydist = yclump - data1[i,1]
        zdist = zclump - data1[i,2]
        dist = np.sqrt(xdist**2.0 + ydist**2.0 + zdist**2.0)

        # Plot approximatiions
        if (dist < 10.0):
            if (random.random() > 0.985):
                plt.scatter(dist, data1[i,4]*data1[i,3]*units, s=20.0, marker='o', facecolor='none', edgecolor='green')
                plt.scatter(dist, data1[i,5]*data1[i,3]*units, s=20.0, marker='s', facecolor='none', edgecolor='red')
                plt.scatter(dist, data1[i,6]*data1[i,3]*units, s=20.0, marker='*', facecolor='none', edgecolor='blue')
                #plt.scatter(dist, data1[i,3]*units, s=20.0,marker='^',facecolor='none',edgecolor='black') # test
                #print(i,dist)

    # Plot single point for each approximation, to be used in legend
    plt.scatter(0.0, 0.0, s=20.0, marker='o', facecolor='none', edgecolor='green', label='Stamatellos')
    plt.scatter(0.0, 0.0, s=20.0, marker='s', facecolor='none', edgecolor='red', label='Lombardi')
    plt.scatter(0.0, 0.0, s=20.0, marker='*', facecolor='none', edgecolor='blue', label='Combined')

    # Plot 2000 particles from full SPH run
    #plt.plot(clump_rad,clump_rho,color='black',lw=2.0)
    plt.plot(cyl_rad[0:2000], cyl_surfcoldens[0:2000], color='black', ls='dashed', lw=2.0, label='SPH')

    # Formatting
    plt.xlim(0.01, 10.0)
    plt.ylim(0.1, 100000)
    plt.yscale('log')
    plt.xlabel('Clump radius (AU)')
    #plt.ylabel('Density (g cm$^{-3}$)')
    plt.ylabel('Column density (g cm$^{-2}$)')
    plt.legend(loc='upper right')
    plt.savefig('Clump_column_density.png')
    plt.show()


def vis_3D(clump_x, clump_y, clump_z):
    """
    Plots the clump as 3d scatter using VisPy
    (NOTE: must be run from terminal, not IPython [unless the user knows how to use vispy using IPython])
    In:
        > clump_x, clump_y, clump_z - (arrs) particle positions in the whole clump (1 per shell)
    """
    print("Initialising plot")
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    pos = np.column_stack([clump_x, clump_y, clump_z])

    print("Adding data to scatter")
    scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, .5), size=1, symbol='o')

    print("Adding view to scatter and finishing up")
    view.add(scatter)
    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()


def vis_3D_clump_in_all(all_x, all_y, all_z, clump_x, clump_y, clump_z):
    """
    Show the clump in the context of all the data
    (NOTE: must be run from terminal, not IPython [unless the user knows how to use vispy using IPython])
    In:
        > all_x, all_y, all_z - (arrs) particle positions in the whole simulation
        > clump_x, clump_y, clump_z - (arrs) particle positions in the whole clump (1 per shell)
    """
    # Setup
    print("Initialising plot")
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()

    # Data prep
    pos_clump = np.column_stack([clump_x, clump_y, clump_z])
    col_clump = np.array([[1, 0, 0]] * len(pos_clump))

    # Filtering clump out of all data
    mask_x = np.in1d(all_x, clump_x)
    mask_y = np.in1d(all_y, clump_y)
    mask_z = np.in1d(all_z, clump_z)
    mask_combined = np.logical_and.reduce((mask_x, mask_y, mask_z))
    pos_all_sub = np.column_stack([all_x, all_y, all_z])[~mask_combined]
    col_all_sub = np.array([[1, 1, 1]] * len(pos_all_sub))

    # Adding to plot
    print("Adding data to scatter")
    pos = np.vstack((pos_all_sub, pos_clump))
    colors = np.vstack((col_all_sub, col_clump))
    scatter.set_data(pos, edge_width=0, face_color=colors, size=1, symbol='o')

    # Plotting
    print("Adding view to scatter and finishing up")
    view.add(scatter)
    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run() 