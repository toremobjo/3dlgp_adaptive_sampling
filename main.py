from Agent import AUV
from Agent import getValue
import numpy as np
import pandas as pd
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.express as px
import SpatialLogGP
import plotly.graph_objects as go


nv =4
h = AUV(vno=0,no_vs=4)
f = AUV(x_init=200,y_init=100,vno=1,no_vs=4)
r = AUV(x_init=1000,y_init=-500,vno=2,no_vs=4)
t = AUV(x_init=0,y_init=0,vno=3,no_vs=4)
auvs = [h,f,r,t]


run_iterations = 3600
x = []
y = []
z = []
t = []

for run in range(run_iterations):
    for i, ag in enumerate(auvs):
        if not ag.get_state():
            ag.wp_update_iteration += 1
            print(int(100*ag.update_iteration/run_iterations), "Percent done")
            if not ag.init:
                ag.init = True
                ag.prior = True
                ag.set_wp(ag.prior_wps[0])

            elif ag.prior and ag.wp_update_iteration<len(ag.prior_wps):
                ag.set_wp(ag.prior_wps[ag.wp_update_iteration])

            elif ag.prior and ag.wp_update_iteration == len(ag.prior_wps):
                ag.prior = False
                ag.adapting = True
                ag.segment()
                df = ag.df
                for j in range(nv):
                    if auvs[j] != ag:
                        auvs[j].ingest_df(df)
                        auvs[j].ingest_pos(ag.get_xyz(),ag.vehicle_no)
                ag.set_wp(ag.adapt())
            elif ag.adapting:
                ag.segment()
                df = ag.df
                for j in range(nv):
                    if auvs[j] != ag:
                        auvs[j].ingest_pos(ag.get_xyz(), ag.vehicle_no)
                        auvs[j].ingest_df(df)
                ag.set_wp(ag.adapt())

        ag.update()
        xx = ag.get_xyz()
        if i==0:
            x.append(xx[0])
            y.append(xx[1])
            z.append(xx[2])
            t.append(run)

df = pd.DataFrame(columns=["x", "y", "z","auv"])
for i,ag in enumerate(auvs):
    auvname = []
    for j in range(len(ag.xl)):
        auvname.append(str(i))
    tdf = pd.DataFrame({"x":ag.xl,"y":ag.yl,"z":ag.zl,"auv":auvname})
    df = pd.concat([df,tdf])

fig = px.line_3d(df, x="x", y="y", z="z", color='auv')
fig.show()

ss = np.array(auvs[0].scorel,dtype=object)
plt.plot(ss.T[(4,)],ss.T[(0,)],label = "Total score ")
plt.plot(ss.T[(4,)],ss.T[(1,)],label= "Predictive mean")
plt.plot(ss.T[(4,)],ss.T[(2,)],label= "Uncertainty ")
plt.plot(ss.T[(4,)],ss.T[(3,)],label= "Avoidance")
plt.legend()
plt.suptitle("Adaptive waypoint score for vehicle #0")
plt.show()


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


for ii in range(nv):
    ss = np.array(auvs[ii].scorel,dtype=object)

    rmse = []
    times = []
    for i, log in enumerate(ss):
        sim = []

        for j, g in enumerate(grid):
            sim.append((getValue(g[0],g[1],g[2],time=log[4],noise=False)-log[5][j])**2)
        rmse.append(np.sqrt(np.mean(sim)))
        times.append(log[4])

    plt.plot(times,rmse)
    with open('log2.npy', 'wb') as f:
        np.save(f, ss)

for auv in auvs:
    pf, pc = auv.gp.evaluate(auv.df)
    fig = go.Figure(data=[go.Volume(
            x=grid.T[0],
            y=grid.T[1],
            z=-grid.T[2],
            value=pf,
            isomin=0.1,
            isomax=3.8,
            opacity=0.5,  # needs to be small to see through all surfaces
            surface_count=20,  # needs to be a large number for good volume rendering
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
        )])
    fig.show()


