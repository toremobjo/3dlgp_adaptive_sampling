#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.spatial as scip
from datetime import datetime
import matplotlib.pyplot as plt
# for development
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GriddedLogGaussianProcess3D:
    def __init__(self,lscales,grid_size,grid_extent,sigma2,kernel="SE",time_sat = 10800, nugget=0.01, ):

        self.ndim = 3                       # number of dimensions
        self.lscales = lscales              # length scale for each ndim
        self.grid_size = grid_size          # number of nodes in grid for each of ndim
        self.grid_extent = grid_extent      # extent of each grid dimension in meters
        self.kernel = kernel                # Wich kernel to use
        self.time_sat = time_sat            # Saturation time for measurements, for added variance of old measurements.
        self.nugget = nugget                # Nugget effect variance
        self.sigma2 = sigma2                # Sigma squared, variance of all data
        self.nogrid = grid_size[0]*grid_size[1]*grid_size[2]

        e = []
        g = []
        for i in range(0,self.ndim):
            e.append(np.linspace(0,grid_extent[i],num=grid_size[i]+1))
            g.append(np.linspace(grid_extent[i]/(grid_size[i]*2),grid_extent[i]-grid_extent[i]/(grid_size[i]*2),num=grid_size[i]))

        self.grid = []
        self.gridno = []
        i=0
        for x in g[0]:
            for y in g[1]:
                for z in g[2]:
                    self.grid.append([x,y,z])
                    self.gridno.append(i)
                    i+=1

        self.F = np.zeros(len(self.grid))  # Design matrix, F_ikj = 1 for all cells containing measurements
        self.grid = np.array(self.grid)
        self.gridno = np.array(self.gridno)

        #Generate covariance matrix
        if self.kernel == "SE":
            self.cov = self.get_se_cov(self.lscales,self.sigma2,self.grid)
        else:
            raise Exception('Implement your own GD kernel')
        print("GP Initialized")

    def get_se_cov(self,lscales,sigma2,sites):
        xx = sites.T[0]/lscales[0]
        yy = sites.T[1]/lscales[1]
        zz = sites.T[2]/lscales[2]
        xa = xb = np.array((xx,yy,zz)).T
        sqnorm = -0.5 * scip.distance.cdist(xa,xb,"sqeuclidean")
        cov =  sigma2*np.exp(sqnorm)
        return cov

    def update_f(self,data):
        self.F = np.zeros(len(self.grid),dtype=bool)
        c = data["c"].values
        self.F[c.astype(int)]=True

    def evaluate(self,data):
        t_now = max(data["t"])
        self.update_f(data)
        dval = np.log(data["d"].values)
        X = np.array([np.ones(self.nogrid), self.grid.T[0], self.grid.T[1], self.grid.T[2]]).T[self.F] #, self.grid.T[0], self.grid.T[1], self.grid.T[2]
        b = np.linalg.lstsq(X, dval, rcond=-1)[0]
        mu = b[0]*np.zeros( self.nogrid) #+ b[1]*self.grid.T[0] + b[2]*self.grid.T[1] +b[3]*self.grid.T[2]
        d_d = dval - mu[self.F]

        ages = []
        for gn in self.gridno[self.F]:
            ages.append(t_now-data[data["c"]==gn]["t"].values/self.time_sat)
        ages = np.array(ages)

        tau = np.diag(self.sigma2*(1.0-np.exp(-ages))) + self.nugget**2*np.eye(sum(self.F),sum(self.F))
        k_bb = self.cov[self.F].T[self.F] + tau
        k_sb = self.cov[:,self.F]

        inv_kbb = np.linalg.inv(k_bb)
        pred_field = k_sb @ inv_kbb @ d_d + mu
        resulting_cov = self.cov - k_sb @ inv_kbb @ k_sb.T

        ## Update to log

        pred_field = np.exp(pred_field+0.5*np.diag(resulting_cov))
        resulting_cov = np.square(pred_field)*(np.exp(resulting_cov)-1)

        # FOR development


        return pred_field,resulting_cov
