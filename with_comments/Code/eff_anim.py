
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# EFF ANNIMATION
# #####################################################################################################################

fig, ax = plt.subplots(figsize = (5, 8))

plt.xlabel('Efficiency (%)')

plt.ylabel('')

ax.set(ylim = (0.2,1.6))

ax.set(xlim = (1.1 * np.max(efficiency), 0))

#ax.spines['bottom'].set_position('zero')

plt.rcParams.update({'font.size': 16})

plt.tight_layout()

V = np.squeeze(iv_curve[0, :, :, 0])
### OLAYA ### iv_curve[0,:,:,i]: current density in the I-V characteristics, V values 
### OLAYA ### removes axes of length one from iv_curve[0,:,:,0]

line = ax.plot(np.squeeze(efficiency[i,:,:]), np.squeeze(E_THERM), lw = 3)[0]
### OLAYA ### removes axes of length one from efficiency[i,:,:]
### OLAYA ### removes axes of length one from E_THERM

def animate(i):

    line.set_xdata(np.squeeze(efficiency[i,:,:]))
    ### OLAYA ### set_xdata() dynamic update of the data in the x axis
    ### OLAYA ### removes axes of length one from efficiency[i,:,:]

    line.set_ydata(np.squeeze(E_THERM))
    ### OLAYA ### set_ydata() dynamic update of the data in the y axis
    ### OLAYA ### removes axes of length one from E_THERM

    print("\r {}".format(np.round((i + 1) / NUM_TIME_STEPS * 100, decimals = 2)), '%', end="")
    ### OLAYA ### Check this out?

    plt.rcParams.update({'font.size': 16})

anim = FuncAnimation(fig, animate, interval = 100, frames = 100)
### OLAYA ### FuncAnimation() makes an animation by repeatedly calling a function

anim.save('tmp.mp4', writer = 'ffmpeg', fps = 25, dpi = 150)

plt.draw()
### OLAYA ### draw() redraws the current figure, updates the figure that has been altered but not re-drawn

plt.show()

# #####################################################################################################################
