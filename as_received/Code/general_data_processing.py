# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:39:50 2019

@author: Richard

general_data_processing can be used to visualize many of the data created by 
simulation.py. Therefore, this file can only be run, when first simulation.py 
is used and created the data. 
"""
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

#%%
# =============================================================================
# Plot a single contour plot
# =============================================================================

def plot_contour(x, y, f_xy):
    # Editing the colormap 
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0:50] = []
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize = (10, 7))
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    
    contour_opts = {'linewidths': 2.5, 'colors':'k'}
    cs = ax.contour(x, y, np.transpose(f_xy), ** contour_opts)
    ### OLAYA ### next line modified due to suggestions from python warnings
    #mappable = ax.pcolormesh(x, y, np.transpose(f_xy), vmin = 0, vmax = np.max(f_xy), cmap = cmap)
    mappable = ax.pcolormesh(x, y, np.transpose(f_xy), vmin = 0, vmax = np.max(f_xy), cmap = cmap, shading = 'auto')
    cax.cla()
    
    fig.colorbar(mappable, cax = cax)
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.weight':'bold'})
    
    ax.set_xlabel('Bandgap energy (eV)', weight = 'bold')
    ax.set_ylabel('Thermodynamic potential (V)', weight = 'bold')
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width = 4)
    ax.yaxis.set_tick_params(width = 4)
    ax.clabel(cs, inline = 2, fontsize = 18, fmt = '%1.0f')
    plt.draw
    plt.savefig("tmp.svg", format = "svg")

### OLAYA ### Next line has been modified as per Richard indications
#plot_contour(E_G / Q, E_THERM, np.squeeze(efficiency[:, :, DAYTIME == 15]))
plot_contour(E_G / Q, E_THERM, np.squeeze(efficiency[:, :, 0]))

plt.show()

# =============================================================================

#%% Plot a (parameter-dependent) IV-curve, make sure you only have one E_G
# =============================================================================

cmap = cm.get_cmap('Blues', NUM_TIME_STEPS+1)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(NUM_TIME_STEPS):
    plt.plot(np.squeeze(iv_curve[0, :, :, i]),
             np.squeeze(iv_curve[1, :, :, i]) * 1000, 
             label = 'IV curve', color = cmap(i + 1),lw = 3)

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

# =============================================================================
#%% Plot a (parameter-dependent) STC vs E_THERM graph, 
# make sure you only have one E_G
# =============================================================================

cmap = cm.get_cmap('Blues', NUM_TIME_STEPS)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(NUM_TIME_STEPS):
    plt.plot(E_THERM,
             np.squeeze(efficiency[:, :, i]), 
             label = 'IV curve', color = cmap(i),lw = 3)

plt.xlim((0,0.7))
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

# =============================================================================
#%% Plot a (parameter-dependent) STC vs Daytime graph, 
# make sure you only have one E_G and one E_THERM
# =============================================================================

### OLAYA ### Error as daytime not defined yet
plt.plot(DAYTIME, np.squeeze(output_power))

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

#%%
# =============================================================================
# Animate a parameter/time-dependent iv_curve (make sure to use only one E_G)
# If you want to plot multiple bandgaps, the contour animator below is recom-
# mended
# =============================================================================

fig, ax = plt.subplots(figsize = (8, 6))
plt.xlabel('Voltage vs NHE (V)')
plt.ylabel('j (mA/cm$^2$)')
plt.grid()

ax.set(ylim = (-20,20))
ax.set(xlim = (0.5, 1.8))
ax.spines['bottom'].set_position('zero')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

V = np.squeeze(iv_curve[0, :, :, 0])
line = ax.plot(np.squeeze(iv_curve[0,:,:,0]), 
               1000 * np.squeeze(iv_curve[1,:,:,0]), lw = 3)[0]

plt.legend()
def animate(i):
    line.set_ydata(1000 * np.squeeze(iv_curve[1,:,:,i]))
    line.set_xdata(np.squeeze(iv_curve[0,:,:,i]))
    print("\r {}".format(np.round((i + 1) / NUM_TIME_STEPS * 100,decimals = 2)),
          '%', end = "")


    plt.rcParams.update({'font.size': 16})

anim = FuncAnimation(fig, animate, interval = 100, frames = NUM_TIME_STEPS)
anim.save('tmp.mp4', writer = 'ffmpeg', fps = 20, dpi = 300)
plt.draw()
plt.show()

#%%
# =============================================================================
# Animate a parameter/time-dependent contourplot. If you want to animate the 
# output power instead of the efficiency, replace efficiency with output_power
# =============================================================================

### ERROR as deytime is not defined
fig, ax = plt.subplots(figsize = (10, 7))
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
# Change the levels to your own convenience for clearer contour-plots
contour_opts = {'levels': np.linspace(0, 36, 13), 'lw': 2, 'colors':'k'}
contour_opts2 = {'levels': np.linspace(0, 36, 13)}

plt.rcParams.update({'font.size': 14})
plt.xlabel('Bandgap energy (eV)')
plt.ylabel('Thermodynamic potential (V)')

START_TIME = time.time()
cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0:50] = []
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, cmap.N)

def animate(i):
     ax.collections = []    
#    cax2 = ax.contour(E_G/Q, E_THERM, 
#                       np.transpose(efficiency[:, :, i]), **contour_opts)

     ax.set_xlabel('Bandgap energy (eV)')
     ax.set_ylabel('Thermodynamic potential (V)')
     ax.set_title('Time of day =' + str.format('{0:.1f}', (DAYTIME[i])) + 'h')
     mappable = ax.contourf(E_G/Q, E_THERM, np.transpose(output_power[:, :, i]), vmin = 0, vmax = np.max(output_power), cmap = cmap, ** contour_opts2)
     cax.cla()
     fig.colorbar(mappable, cax = cax)
     
     print("\r {}".format(np.round((i) / NUM_TIME_STEPS * 100,decimals = 2)),'%', end = "")

        
anim = FuncAnimation(fig, animate, interval = 100, frames = NUM_TIME_STEPS)
anim.save('temporary_3.mp4', writer = 'ffmpeg', fps = 30, dpi = 300)
ELAPSED_TIME = time.time() - START_TIME
plt.rcParams.update({'font.size': 14})
print('\n Done! The elapsed time = ', ELAPSED_TIME)

plt.show()

