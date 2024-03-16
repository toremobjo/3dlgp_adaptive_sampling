import matplotlib.pyplot as plt
import numpy as np
from SpatialLogGP import GriddedLogGaussianProcess3D
import time
import pandas as pd
import plotly.graph_objects as go
from simulator_cf import getValue



position = [78.93295, 11.953366]
origin_rad = [position[0]*np.pi/180,position[1]*np.pi/180]

gp_o = [78.93295178*np.pi/180.0,11.95336597*np.pi/180.0]
grid_extent = [3000, 3000, 50]
grid_size = [30, 30, 30]
orientation = -45.0*np.pi/180.0
ndim = len(grid_extent)
e = []
g = []

for i in range(0, ndim):
    e.append(np.linspace(0, grid_extent[i], num=grid_size[i] + 1))
    g.append(np.linspace(grid_extent[i] / (grid_size[i] * 2), grid_extent[i] - grid_extent[i] / (grid_size[i] * 2),
                         num=grid_size[i]))

grid = []
gridd = []
gridno = []
i = 0
for x in g[0]:
    for y in g[1]:
        for z in g[2]:
            grid.append([x, y, z])
            gridno.append(i)
            i += 1

data = []
ww = 100
times = np.linspace(0,24*3600,num=ww)

for i in range(ww):
    for g in grid:
        data.append(getValue(g[0],g[1],g[2],time=times[i]))
        grid = np.array(grid)

    fig = go.Figure(data=[go.Volume(
            x=grid.T[0],
            y=grid.T[1],
            z=-grid.T[2],
            value=np.log(data),
            isomin=-1.1,
            isomax=2.5,
            opacity=0.5,  # needs to be small to see through all surfaces
            surface_count=40,  # needs to be a large number for good volume rendering
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
        )])

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.2),
        eye=dict(x=0.5+(i/ww) , y=-1.2+(i/ww) , z=.5)
    )
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1, z=0.5), scene_camera=camera)
    fig.write_image("fig/gifinput/inp" + str(int(times[i])) + ".png")
    print(i)
    data = []

fig.show()

