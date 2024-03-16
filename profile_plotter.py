from simulator_cf import getValue
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

grid_extent = [3000, 3000, 50]
grid_size = [10, 10, 10]
orientation = -45.0*np.pi/180.0
ndim = len(grid_extent)
e = []
g = []

for i in range(0, ndim):
    e.append(np.linspace(0, grid_extent[i], num=grid_size[i] + 1))
    g.append(np.linspace(grid_extent[i] / (grid_size[i] * 2), grid_extent[i] - grid_extent[i] / (grid_size[i] * 2),
                         num=grid_size[i]))

tidalcycle = 44700
tidetime = np.linspace(0,tidalcycle*1.05,num=1000)
tide = 2.1+1.5*np.sin(2*np.pi*tidetime/tidalcycle)


times = np.linspace(0,tidalcycle,num=6)
depths = np.linspace(0,50,num=1000)

colors = ["b","orange","g","r","purple","brown"]

for x in g[0]:
    for y in g[1]:
        fig, ax = plt.subplots(2,1,figsize = (8,6),gridspec_kw={'height_ratios': [3, 1]})
        ax[1].plot(tidetime, tide, c="black",label = "Tide [m]")
        for i,t in enumerate(times):
            vs = []
            for z in depths:
                vs.append(getValue(x, y, z,time=t,noise=False)+5*i)
            ax[0].plot(vs,depths,label = str(int(np.floor(t/3600))) + "h " + str(int((t%3600)/60)) + "m",c=colors[i])
            ax[0].axvline(5*i,c="black",alpha=0.5)
            ax[0].axvline(5 * i+1, c="black", alpha=0.15)
            ax[0].axvline(5 * i+2, c="black", alpha=0.15)
            ax[0].axvline(5 * i + 3, c="black", alpha=0.15)
            ax[0].axvline(5 * i + 4, c="black", alpha=0.15)
            ax[1].axvline(t,c=colors[i],alpha=0.99)

        ax[0].invert_yaxis()
        ax[0].set_xticks([0,2,5,7,10,12,15,17,20,22,25,27],["0","2","0","2","0","2","0","2","0","2","0","2"])
        ax[0].set_xlabel("Simualted chlorophyll a profiles [$\mu$g/L]")
        ax[0].set_ylabel("Depth [m]")
        ax[0].legend(loc="lower right")
        ax[0].xaxis.tick_top()

        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("tide [m]")
        ax[1].legend(loc="upper right")
        plt.savefig("fig/profiles/profilesx" + str(int(x)) + "y" + str(int(y)) + ".png",dpi=300)
        plt.show()


