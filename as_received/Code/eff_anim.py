# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:44:21 2019

@author: Richard
"""


fig, ax = plt.subplots(figsize = (5, 8))
plt.xlabel('Efficiency (%)')
plt.ylabel('')

ax.set(ylim = (0.2,1.6))
ax.set(xlim = (1.1 * np.max(efficiency), 0))
#ax.spines['bottom'].set_position('zero')

plt.rcParams.update({'font.size': 16})
plt.tight_layout()

V = np.squeeze(iv_curve[0, :, :, 0])
line = ax.plot(np.squeeze(efficiency[i,:,:]), 
               np.squeeze(E_THERM), lw = 3)[0]

def animate(i):
    line.set_xdata(np.squeeze(efficiency[i,:,:]))
    line.set_ydata(np.squeeze(E_THERM))
    print("\r {}".format(np.round((i + 1) / NUM_TIME_STEPS * 100, decimals = 2)), '%', end="")


    plt.rcParams.update({'font.size': 16})

anim = FuncAnimation(fig, animate, interval = 100, frames = 100)
anim.save('tmp.mp4', writer = 'ffmpeg', fps = 25, dpi = 150)
plt.draw()
plt.show()
