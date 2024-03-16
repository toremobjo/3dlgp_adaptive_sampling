import matplotlib.pyplot as plt
import numpy as np

dist = np.linspace(0,5,num=100)
experimental = np.linspace(0,4,num=10)
nugg = 0.2
scale = 1.0
lscale = 1.0

exponential = nugg + scale*(1-np.exp(-dist/lscale))
matern      = nugg + scale*(1-(1+dist*np.sqrt(3))*(np.exp(-dist*np.sqrt(3)/lscale)))
cauchy      = nugg + scale*(1-(1/((1+dist/lscale)**3)))
gauss       = nugg + scale*(1-np.exp(-(dist/lscale)**2))

exout = nugg + scale*(1-np.exp(-(experimental/lscale)**2))  + np.random.normal(0.0,0.05,10)

fig, ax = plt.subplots(1,1,figsize = (6,4))

ax.plot(dist,exponential,c="black",label = "Exponential")
ax.plot(dist,matern,c="black",linestyle = "dotted", label = "Matern 3/2")
ax.plot(dist,cauchy,c="black",linestyle = "dashed", label = "Cauchy type")
ax.plot(dist,gauss,c="black",linestyle = "dashdot", label = "Gaussian / Squared exponential")
#ax.scatter(experimental,exout,s = 100,edgecolor="black",facecolor = "none", label ="Experimental data")

ax.axvline(x=lscale,c="r",label = "Characteristic length")
ax.axhline(y=nugg,c="r",linestyle = "dotted",label = "Nugget")
ax.axhline(y=nugg+scale,c="r",linestyle = "dashed",label = "Sill")

ax.set_ylim(0.0,(scale+nugg)*1.1)
ax.set_xlim(0,3)
ax.legend()
ax.set_ylabel("Semivariance $\sigma^2$")
ax.set_xlabel("Distance")
plt.savefig("kernels.png",dpi = 400)
plt.show()
