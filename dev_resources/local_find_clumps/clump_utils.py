import numpy as np

# code_L = 1.496e13 # Code Length unit, in cm
code_L = 1 # code units

def find_centre(data):
    """
    Finds the centre of the densest clump between 50-100 au.
    In:
        > data - (arr) input SPH data array
    Out:
        > max_dens - (float) maximum density value
        > rclump - (float) radius of clump
        > xclump, yclump, zclump - (floats) x, y, z-coords of clump centre
    """
    # Finds maximum density (centre?)
    print("Finding centre")
    max_dens = 0.0

    # For every particle
    for i in range(np.shape(data[:,0])[0]):
        # Finding particle's radial distance
        drad = np.sqrt(data[i,0]**2.0 + data[i,1]**2.0)

        # If radial distance is between 50 and 100 AU and away from the disc midplane
        if ((drad > 50.0) and (drad < 100.0) and (data[i,0] > 0.0)):

            # If density of particle is greater than current max, update max
            if (data[i,5] > max_dens):
                max_dens = data[i,5]
                rclump = drad
                xclump = data[i,0]
                yclump = data[i,1]
                zclump = data[i,2]

    print("The centre is at:")
    print(max_dens,rclump,xclump,yclump,zclump)
    return max_dens, rclump, xclump, yclump, zclump


def find_clump_coldens(data, xclump, yclump, zclump):
    """
    Finds the shell column densities.
    In:
        > data - (arr) input SPH data array (already in physical units)
        > xclump, yclump, zclump - (floats) x, y, z-coords of clump centre
    Out:
        > clump_rho - (arr) clump densities at each shell
        > clump_rad - (arr) radial distance of each shell
        > clump_num - (arr) no. of particles in each shell
        > clump_coldens - (arr) column density of each shell
        > clump_x, clump_y, clump_z - (arrs) particle positions in the whole clump (1 per shell)
    """
    N_shell = 20000 # Number of shell to sample
    R_clump = 1.0 # Max clump radius
    # Clump values
    print("Finding clump particle properties")
    clump_rho = np.zeros(N_shell)
    clump_rad = np.zeros(N_shell)
    clump_num = np.zeros(N_shell)
    clump_x = np.zeros(N_shell)
    clump_y = np.zeros(N_shell)
    clump_z = np.zeros(N_shell)

    # Iterate over all particles
    for i in range(np.shape(data[:,0])[0]):
        # Distance of all particles from the centre of clump
        xdist = xclump - data[i,0]
        ydist = yclump - data[i,1]
        zdist = zclump - data[i,2]
        dist = np.sqrt(xdist**2.0 + ydist**2.0 + zdist**2.0)

        # If distance is within 10au, add particle to clump array
        if (dist < R_clump):
            n = np.int_(dist*N_shell/R_clump) # Up to distance and we want N_part so we multiply distance by 200 to get max_n=N_part
            clump_rho[n] += data[i,5] # Add ith particle density to appropriate cell
            clump_rad[n] = n * R_clump / N_shell # radial distance, essentially just 'dist'
            clump_num[n] += 1 # Increase number of particles in n-th radius by one
            clump_x[n] = data[i,0] # Store particle posns (this will get overwritten with last particle posn)
            clump_y[n] = data[i,1]
            clump_z[n] = data[i,2]
        # print(i,dist,n)

    # Density calculation
    print("Calculating densities at each radius")
    for i in range(N_shell):
        if (clump_num[i] > 0):
            clump_rho[i] /= clump_num[i] # get density by dividing integrated rho by # of parts.
        # print(i,clump_num[i],clump_rho[i])

    # Column density calculation
    print("Calculating column densities")
    clump_coldens = np.zeros(N_shell)
    dr = R_clump * code_L / np.float_(N_shell) # Distance step [cm]
    for i in range(N_shell):
        for j in range(i, N_shell):
            clump_coldens[i] += clump_rho[j] * dr # At each radius integrate all the densities above it to get column density
            #print(i,clump_rad[i],clump_coldens[i])
    return clump_rho, clump_rad, clump_num, clump_coldens, clump_x, clump_y, clump_z