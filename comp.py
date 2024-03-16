import numpy as np
from simulator_cf import getValue
import matplotlib.pyplot as plt


gp_o = [78.93295178*np.pi/180.0,11.95336597*np.pi/180.0]
grid_extent = [3000, 3000, 50]
grid_size = [15, 15, 10]
orientation = -45.0*np.pi/180.0
ndim = len(grid_extent)
e = []
g = []

for i in range(0, ndim):
    e.append(np.linspace(0, grid_extent[i], num=grid_size[i] + 1))
    g.append(np.linspace(grid_extent[i] / (grid_size[i] * 2), grid_extent[i] - grid_extent[i] / (grid_size[i] * 2),
                         num=grid_size[i]))

grid = []
gridno = []
i = 0
for x in g[0]:
    for y in g[1]:
        for z in g[2]:
            grid.append([x, y, z])
            gridno.append(i)
            i += 1


ww = 100
times = np.linspace(0,24*3600,num=ww)
rmses = []

for jj in range(100):
    data = []
    rmse = []
    for i, t in enumerate(times):
        sim = []
        for j, g in enumerate(grid):
            sim.append((getValue(g[0],g[1],g[2],time=t,noise=False)-(getValue(g[0],g[1],g[2],time=t,noise=True)))**2)
        rmse.append(np.sqrt(np.mean(sim)))
    print(jj)
    rmses.append(rmse)

rmses = np.array(rmses)
rmsevar = []#np.array(rmses).var()

avg = []
avgm = []
avgp = []
var = []
i = []
for it, rr in enumerate(rmses.T):
    avg.append(np.mean(rr))
    var.append(np.sqrt(np.var(rr)))
    avgm.append(np.mean(rr) - np.sqrt(np.var(rr)))
    avgp.append(np.mean(rr) + np.sqrt(np.var(rr)))
    i.append(it)
plt.plot(times,avg)
plt.fill_between(times, avgm,avgp,alpha=0.3)
plt.savefig("RMSE2.png",dpi=300)
plt.show()

print("AAAAAAAAAQAA")