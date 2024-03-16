import numpy as np
import matplotlib.pyplot as plt
from simulator_cf import getValue
import skgstat

x = np.linspace(0,2000,num=15)
y = np.linspace(0,2000,num=15)
z = np.linspace(0,50,num=15)

data = []
xs = []
ys = []
zs = []
ss = []
dd = []
draw = []

for zz in z:
    dd = []
    for xx in x:
        for yy in y:
            dd.append(np.log(getValue(xx,yy,zz)))
            xs.append(xx)
            ys.append(yy)
            zs.append(zz)
            ss = np.sqrt(xx**2 +yy**2)
    data.append(np.array(dd)-np.mean(dd))
    draw.append(np.array(dd))

data = np.array(data).flatten()
draw = np.array(draw).flatten()
pos = np.array([xs,ys]).T
'''
vparam = []
vg = skgstat.Variogram(pos,draw,estimator="cressie",model="matern",use_nugget=True,n_lags=20)
vparam.append(vg.parameters)
print(vg.parameters)
vg.plot().savefig("variogram_b.png",dpi=600)
plt.show()

data = []
xs = []
ys = []
zs = []
ss = []
dd = []
draw = []
x = np.linspace(0,2000,num=10)
y = np.linspace(0,2000,num=10)
z = np.linspace(0,50,num=10)

for xx in x:
    for yy in y:
        dd = []
        for zz in z:
            dd.append(np.log(getValue(xx,yy,zz)))
            zs.append(zz)
        data.append(np.array(dd)-np.mean(dd))
        draw.append(np.array(dd))

data = np.array(data).flatten()
draw = np.array(draw).flatten()
pos = np.array([zs]).T

vparam = []
vg = skgstat.Variogram(pos,draw,estimator="cressie",model="matern",use_nugget=True,n_lags=20)
vparam.append(vg.parameters)
print(vg.parameters)
vg.plot().savefig("variogram_b.png",dpi=600)
plt.show()

print(np.var(draw))
'''
s = []
d = []
for i, posa in enumerate(pos):
    for j, posb in enumerate(pos):
        s.append(np.linalg.norm(posa-posb))
        d.append((data[i]-data[j])**2)


ds = np.array([d,s])
ind = np.argsort(s)
s = np.array(s)[ind]
d = np.array(d)[ind]
bins = 10

vvar = []
bin = []
for i in range(0,bins):
    dd = d[0:int(np.floor(i*len(d)/bins))]
    vvar.append(np.mean(dd)/2)
    bin.append(int(np.floor(i*max(s)/bins)))
plt.scatter(bin,vvar)
plt.show()

print(np.var(data))
print("DONE")

