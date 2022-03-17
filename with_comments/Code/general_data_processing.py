
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# GENERAL DATA PROCESSING 
# #####################################################################################################################

"""
general_data_processing can be used to visualize many of the data created by simulation.py. 
Therefore, this file can only be run, when first simulation.py is used and created the data. 
"""

# #####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.colors

from simulation import E_THERM

'''
from physical_constants import Q
from model_settings import NUM_E_THERM, NUM_TIME_STEPS
from simulation import E_G, E_THERM, DAYTIME, iv_curve, efficiency, output_power
'''

# #####################################################################################################################
# Plot a single contour plot 
# #####################################################################################################################

def plot_contour(x, y, f_xy):

    # Editing the colormap 

    cmap = plt.cm.viridis
    ### OLAYA ### set the color map to the viridis color palette

    cmaplist = [cmap(i) for i in range(cmap.N)]
    ### OLAYA ### N: linearly normalizes data between 0 and 1

    cmaplist[0:50] = []

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, cmap.N)
    ### OLAYA ### mlinearSegmentedColormap() colormals objects based on lookup tables using linear sgments
    ### OLAYA ### from_list() create a LinearSegmentedColormap from a list of colors
    ### OLAYA ### 'mcm' is the name of the colormap
    ### OLAYA ### cmaplist, cmap.N is the colors array-like of colors or array-like of (value,color)
    ### OLAYA ### N is the number of rgb quantization levels

    fig, ax = plt.subplots(figsize = (10, 7))
    
    div = make_axes_locatable(ax)
    ### OLAYA ### make_axes_locatable() takes and axisting axes, adds it to a new axesdivider and returns this

    cax = div.append_axes('right', '5%', '5%')
    ### OLAYA ### append.axes() can be used to create a new axes of a given side of the original axes
    
    contour_opts = {'linewidths': 2.5, 'colors':'k'}
    
    cs = ax.contour(x, y, np.transpose(f_xy), ** contour_opts)
    ### OLAYA ### Trying to solve a warning:
    #cs = ax.contour ( [ [0, 0], [0, x] ], [ [0, 0], [0, y] ], [ [0, 0], [0, transpose_f_xy] ], **contour_opts )

    ### OLAYA ### next line modified due to suggestions from python warnings
    #mappable = ax.pcolormesh(x, y, np.transpose(f_xy), vmin = 0, vmax = np.max(f_xy), cmap = cmap)
    mappable = ax.pcolormesh(x, y, np.transpose(f_xy), vmin = 0, vmax = np.max(f_xy), cmap = cmap, shading = 'auto')

    cax.cla()
    ### OLAYA ### cla() clears the current axes
    
    fig.colorbar(mappable, cax = cax)
    ### OLAYA ### colorbar() adds a colorbar to a plot

    plt.rcParams.update({'font.size': 20})
    
    plt.rcParams.update({'font.weight':'bold'})
  
    ax.set_xlabel('Bandgap energy (eV)', weight = 'bold')

    ax.set_ylabel('Thermodynamic potential (V)', weight = 'bold')
   
    for axis in ['top','bottom','left','right']:
    
      ax.spines[axis].set_linewidth(4)

    ax.xaxis.set_tick_params(width = 4)
 
    ax.yaxis.set_tick_params(width = 4)
 
    ax.clabel(cs, inline = 2, fontsize = 18, fmt = '%1.0f')

    plt.draw()

    plt.savefig("tmp.svg", format = "svg")

#### OLAYA ### Next line was recommended by richard and the following one after that, it is the previous one
#plot_contour(E_G / Q, E_THERM, np.squeeze(efficiency[:, :, DAYTIME == 15]))
plot_contour(E_G/Q, E_THERM, np.squeeze (efficiency[:,:,0]))

# #####################################################################################################################
# Plot a (parameter-dependent) IV-curve, make sure you only have one E_G
# #####################################################################################################################

cmap = cm.get_cmap('Blues', NUM_TIME_STEPS+1)
### OLAYA ### get_cmap() gets a colormap instance

fig = plt.figure()

ax = fig.add_subplot(111)

for i in range(NUM_TIME_STEPS):

    plt.plot(np.squeeze(iv_curve[0, :, :, i]),
             np.squeeze(iv_curve[1, :, :, i]) * 1000, 
             label = 'IV curve', color = cmap(i + 1),lw = 3)
    ### OLAYA ### create the plot for each NUM_TIME_STEPS in the same figure

plt.xlim((0,1))

plt.ylim((0,45))
 
plt.rcParams.update({'font.size': 14})

plt.xlabel('Cell Voltage (V)')

plt.ylabel('Current density (mA/cm$^2$)')

plt.tight_layout()

for axis in ['top','bottom','left','right']:

  ax.spines[axis].set_linewidth(2)
  
ax.xaxis.set_tick_params(width = 2)

ax.yaxis.set_tick_params(width = 2)

plt.savefig("overpotential.svg", format = "svg")

plt.show()

# #####################################################################################################################
# Plot a (parameter-dependent) STC vs E_THERM graph, make sure you only have one E_G
# #####################################################################################################################

cmap = cm.get_cmap('Blues', NUM_TIME_STEPS)
### OLAYA ### get_cmap() gets a colormap instance

fig = plt.figure()

ax = fig.add_subplot(111)

for i in range(NUM_TIME_STEPS):

    plt.plot(E_THERM, np.squeeze(efficiency[:, :, i]), label = 'IV curve', color = cmap(i),lw = 3)
    ### OLAYA ### create the plot for each NUM_TIME_STEPS in the same figure
    ### OLAYA ### create a plot with E_THERM and np.squeeze(efficiency[:, :, i])
    ### OLAYA ### removes axes of length one from efficiency[:,:,i]

### OLAYA ### limites where updated to cover any E_THERM value:
plt.xlim((0,1))

plt.ylim((0,30))

plt.rcParams.update({'font.size': 12})

plt.xlabel('Voltage (V)')

plt.ylabel('Current density (mA/cm$^2$)')

plt.tight_layout()

for axis in ['top','bottom','left','right']:

  ax.spines[axis].set_linewidth(2)
  
ax.xaxis.set_tick_params(width = 2)

ax.yaxis.set_tick_params(width = 2)

plt.show()

# #####################################################################################################################
# Plot a (parameter-dependent) STC vs Daytime graph, make sure you only have one E_G and one E_THERM
# #####################################################################################################################

plt.plot(DAYTIME, np.squeeze(output_power))
### OLAYA ### Create a plot with DAYTIME and np.squeeze(output_power)
### OLAYA ### removes axes of length one from output_power

#plt.xlim((0,0.7))

#plt.ylim((0,30))

plt.rcParams.update({'font.size': 16})

plt.xlabel('Time of day (h)')

plt.ylabel('Output power (mW/cm$^2$)')

plt.tight_layout()

for axis in ['top','bottom','left','right']:

  ax.spines[axis].set_linewidth(2)

ax.xaxis.set_tick_params(width = 2)

ax.yaxis.set_tick_params(width = 2)

plt.show()

# #####################################################################################################################
# Animate a parameter/time-dependent iv_curve (make sure to use only one E_G)
# If you want to plot multiple bandgaps, the contour animator below is recommended
# #####################################################################################################################

fig, ax = plt.subplots(figsize = (8, 6))

plt.xlabel('Voltage vs NHE (V)')

plt.ylabel('j (mA/cm$^2$)')

plt.grid()
### OLAYA ### grid() configures the grid lines

ax.set(ylim = (-20,20))

ax.set(xlim = (0.5, 1.8))

ax.spines['bottom'].set_position('zero')

plt.rcParams.update({'font.size': 16})

plt.tight_layout()

V = np.squeeze(iv_curve[0, :, :, 0])
### OLAYA ### iv_curve[0,:,:,i]: current density in the I-V characteristics, V values 
### OLAYA ### removes axes of length one from iv_curve[0,:,:,0]

line = ax.plot(np.squeeze(iv_curve[0,:,:,0]), 1000 * np.squeeze(iv_curve[1,:,:,0]), lw = 3)[0]
### OLAYA ### removes axes of length one from iV_curve[0,:,:,0]
### OLAYA ### removes axes of length one from iV_curve[1,:,:,0]

### OLAYA ### Next line produced: No handles with labels found to put in legend.
### OLAYA ### as per stackoverflow: is better to use ax.legend() than plt.leyend() 
# plt.legend()
ax.legend()

def animate(i):

    line.set_ydata(1000 * np.squeeze(iv_curve[1,:,:,i]))
    ### OLAYA ### set_ydata() dynamic update of the data in the y axis
    ### OLAYA ### removes axes of length one from iv_curve[1,:,:,i]
    ### OLAYA ### iv_curve[1,:,:,i]: current density in the I-V characteristics, I values 

    line.set_xdata(np.squeeze(iv_curve[0,:,:,i]))
    ### OLAYA ### set_xdata() dynamic update of the data in the x axis
    ### OLAYA ### removes axes of length one from iv_curve[0,:,:,i]
    ### OLAYA ### iv_curve[0,:,:,i]: current density in the I-V characteristics, V values 

    print("\r {}".format(np.round((i + 1) / NUM_TIME_STEPS * 100,decimals = 2)), '%', end = "")
   
    plt.rcParams.update({'font.size': 16})
    
anim = FuncAnimation(fig, animate, interval = 100, frames = NUM_TIME_STEPS)
### OLAYA ### FuncAnimation() makes an animation by repeatedly calling a function, animate

anim.save('tmp.mp4', writer = 'ffmpeg', fps = 20, dpi = 300)

plt.draw()
### OLAYA ### draw() redraws the current figure, updates the figure that has been altered but not re-drawn

plt.show()

# #####################################################################################################################
# Animate a parameter/time-dependent contourplot. 
# If you want to animate the output power instead of the efficiency, replace efficiency with output_power
# #####################################################################################################################

fig, ax = plt.subplots(figsize = (10, 7))

div = make_axes_locatable(ax)
### OLAYA ### make_axes_locatable() takes and axisting aces, adds it to a new axesdivider and returns this

cax = div.append_axes('right', '5%', '5%')
### OLAYA ### append.axes() can be used to create a new axes of a given side of the original axes

# Change the levels to your own convenience for clearer contour-plots

contour_opts = {'levels': np.linspace(0, 36, 13), 'lw': 2, 'colors':'k'}
### OLAYA ### set the contour options.It could be introduced into contour(). It is commented below

contour_opts2 = {'levels': np.linspace(0, 36, 13)}
### OLAYA ### set the contour options.IT could be introduced into contour(). It is used below.

plt.rcParams.update({'font.size': 14})

plt.xlabel('Bandgap energy (eV)')

plt.ylabel('Thermodynamic potential (V)')

START_TIME = time.time()

cmap = plt.cm.viridis
### OLAYA ### set the color map to the viridis color palette

cmaplist = [cmap(i) for i in range(cmap.N)]
### OLAYA ### N: linearly normalizes data between 0 and 1

cmaplist[0:50] = []

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, cmap.N)
### OLAYA ### mlinearSegmentedColormap() colormals objects based on lookup tables using linear sgments
### OLAYA ### from_list() create a LinearSegmentedColormap from a list of colors
### OLAYA ### 'mcm' is the name of the colormap
### OLAYA ### cmaplist, cmap.N is the colors array-like of colors or array-like of (value,color)
### OLAYA ### N is the number of rgb quantization levels

def animate(i):

     ax.collections = []  
     #cax2 = ax.contour(E_G/Q, E_THERM, np.transpose(efficiency[:, :, i]), **contour_opts)
     ### OLAYA ### contour() sets how to plot contour lines

     ax.set_xlabel('Bandgap energy (eV)')
    
     ax.set_ylabel('Thermodynamic potential (V)')
     
     ax.set_title('Time of day =' + str.format('{0:.1f}', (DAYTIME[i])) + 'h')
     
     mappable = ax.contourf(E_G/Q, E_THERM, np.transpose(output_power[:, :, i]), 
        vmin = 0, vmax = np.max(output_power), cmap = cmap, ** contour_opts2)
     ### OLAYA ### contourf() sets how to plot filled contours

     cax.ercla()
     ### OLAYA ### cla() clear the current axis

     fig.colorbar(mappable, cax = cax)
     ### OLAYA ### colorbar() adds a colorbar to a plot

     print("\r {}".format(np.round((i) / NUM_TIME_STEPS * 100,decimals = 2)),'%', end = "")
    
anim = FuncAnimation(fig, animate, interval = 100, frames = NUM_TIME_STEPS)

anim.save('temporary_3.mp4', writer = 'ffmpeg', fps = 30, dpi = 300)

ELAPSED_TIME = time.time() - START_TIME

plt.rcParams.update({'font.size': 14})

print('\n Done! The elapsed time = ', ELAPSED_TIME)

# #####################################################################################################################
