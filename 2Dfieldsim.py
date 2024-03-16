#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as scip

class FlatGaussianRandomField:
    def __init__(self,prior,lscales,grid_size,grid_extent,sigma2 = 0.1,kernel="SE", nugget=0.01):

        self.ndim = 2                       # number of dimensions
        self.lscales = lscales              # length scale for each ndim
        self.grid_size = grid_size          # number of nodes in grid for each of ndim
        self.grid_extent = grid_extent      # extent of each grid dimension in meters
        self.kernel = kernel                # Wich kernel to use
        self.nugget = nugget                # Nugget effect variance
        self.sigma2 = sigma2                # Sigma squared, variance of all data
        self.nogrid = grid_size*grid_size
        self.mu = prior

        e = []
        g = []
        for i in range(0,self.ndim):
            e.append(np.linspace(0,grid_extent[i],num=grid_size+1))
            g.append(np.linspace(grid_extent[i]/(grid_size*2),grid_extent[i]-grid_extent[i]/(grid_size*2),num=grid_size))

        self.grid = []
        self.gridno = []
        i=0
        for x in g[0]:
            for y in g[1]:
                self.grid.append([x,y])
                self.gridno.append(i)
                i+=1

        self.F = np.zeros(len(self.grid))  # Design matrix, F_ikj = 1 for all cells containing measurements
        self.grid = np.array(self.grid)
        self.gridno = np.array(self.gridno)

        #Generate covariance matrix
        if self.kernel == "SE":
            self.cov = self.get_se_cov(self.lscales,self.sigma2,self.grid)
            #self.cov = self.getcov(self.grid,self.grid,[self.sigma2,self.lscales])
        else:
            raise Exception('Implement your own GD kernel')
        print("GP Initialized")

    def get_se_cov(self,lscales,sigma2,sites):
        xx = sites.T[0]/lscales
        yy = sites.T[1]/lscales
        xa = xb = np.array((xx,yy)).T
        sqnorm = -0.5 * scip.distance.cdist(xa,xb,"sqeuclidean")
        cov =  sigma2*np.exp(sqnorm)
        return cov

    def getcov(self,d_sites, p_sites, par):
        sig2 = par[0]
        crange = par[1]

        h = -0.5 * scip.distance.cdist(d_sites / crange, p_sites / crange, 'sqeuclidean')
        # sqnorm = -0.5 * scip.distance.cdist(xa, xb, "sqeuclidean")
        cov = sig2 * np.exp(h)
        return cov

    def update_f(self,grid_no):
        self.F = np.zeros(len(self.grid),dtype=bool)
        self.F[grid_no]=True

    def evaluate(self,data,grid_no):
        self.update_f(grid_no)
        d_d = data - self.mu[grid_no]

        tau = np.diag(self.nugget * np.ones(sum(self.F)))
        k_bb = self.cov[self.F].T[self.F] + tau
        k_sb = self.cov[:, self.F]
        k_bs = k_sb.T
        k_ss = self.cov

        invkb = np.linalg.inv(k_bb)
        pred_field = k_sb @ invkb @ d_d + self.mu
        resulting_cov = k_ss - k_sb @ invkb @ k_bs

        self.mu = pred_field
        self.cov = resulting_cov

        return pred_field,resulting_cov


def getTempVal(x,y,noise = True):
    tmean = 8.3435342
    tsin = -0.9 * np.sin(x/1000.0 + np.pi/6) - 0.6 * np.sin(y/1000.0 - np.pi/6)
    tplume = 0.9*np.exp(((-0.2*(x-750)**2)+(-(y-700)**2))/7000)
    if noise:
        return tmean + tsin + tplume + np.random.normal(0.0,0.1) # nugget = 0.1
    else:
        return tmean + tsin + tplume

def segment(values, x, y, gridsize, grid_extent):
    c = []
    for i, v in enumerate(values):
        if (x[i] < 1000.0) and (x[i] > 0.0) and (y[i] > 0.0) and (y[i] < 1000):
            xx = np.floor(x[i] * gridsize / grid_extent[0])
            yy = np.floor(y[i] * gridsize / grid_extent[1])
            gn = xx * gridsize + yy
            c.append(int(gn))

    val = []
    grid_no = []

    for cc in range(gridsize*gridsize):
        try:
            index = c.index(cc)
            if index:
                val.append(np.mean(values[index]))
                grid_no.append(int(cc))
        except:
            pass
    return val, grid_no

gridsize = 50
xx = np.linspace(0,1000, num=gridsize)
yy = np.linspace(0,1000, num=gridsize)


val = []
clean_val = []
xval = []
yval = []

for x in xx:
    for y in yy:
        val.append(getTempVal(x,y))
        clean_val.append(getTempVal(x,y,noise=False))
        xval.append(x)
        yval.append(y)


val = np.array(val)
clean_val = np.array(clean_val)

fig, ax = plt.subplots(2,2,figsize = (10,8))

aaa = ax[0,0].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),np.reshape(val,(gridsize,gridsize)),10)
ax[0,0].set_title("Noisy field with four measurement points")
bbb = ax[1,0].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),np.reshape(clean_val,(gridsize,gridsize)),10)
ax[1,0].set_title("Underlying field")

nugget = 0.1
sill = np.var(val)

class vehicle:
    def __init__(self):
        self.x = 0
        self.y = 0

        self.measurements  = []
        self.xmeasurements = []
        self.ymeasurements = []

        self.speed = 1.5

        self.wp = [10,10]
        self.state = 0

    def update(self):
        self.state = 5 > np.linalg.norm(np.array([self.x-self.wp[0],self.y-self.wp[1]]))

        heading = np.arctan2(self.wp[1]-self.y,self.wp[0]-self.x)
        self.x += self.speed*np.cos(heading)
        self.y += self.speed*np.sin(heading)

        self.measurements.append(getTempVal(self.x,self.y))
        self.xmeasurements.append(self.x)
        self.ymeasurements.append(self.y)


#Make prior from corner measurements
corners = [[50,50],[950,50],[50,950],[950,950]]
cornervalues  =[]
for corner in corners:
    cornervalues.append(getTempVal(corner[0],corner[1]))
corners = np.array(corners)

ax[0,0].scatter(corners.T[0],corners.T[1],c="red", s = 100)


X = np.array([np.ones(len(corners)), corners.T[0], corners.T[1]]).T
b = np.linalg.lstsq(X, cornervalues, rcond=-1)[0]
mu = b[0]*np.ones(gridsize*gridsize) + b[1]*np.array(xval) + b[2]*np.array(yval)
mu_0 = mu
mu = np.reshape(mu,(gridsize,gridsize))

ccc = ax[0,1].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),mu,10)
ax[0,1].set_title("Prior from measurements")

ddd = ax[1,1].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),mu-np.reshape(val,(gridsize,gridsize)),10)
ax[1,1].set_title("Error in prior, $\mu-x$")

fig.colorbar(aaa,ax=ax[0,0])
fig.colorbar(bbb,ax=ax[1,0])
fig.colorbar(ccc,ax=ax[0,1])
fig.colorbar(ddd,ax=ax[1,1])
plt.show()

mission_time = 6000#3600*3
length_scale = 300.0
grid_extent = [1000,1000]

######## RANDOM PATH #########
random = False
if random:
    auv = vehicle()
    model = FlatGaussianRandomField(mu_0, length_scale,gridsize,[1000,1000],sigma2=0.05,kernel="SE", nugget=0.01)
    trackx = [0]
    tracky = [0]
    figno = 0

    for t in range(mission_time):
        auv.update()
        if auv.state:
            vals, gno = segment(auv.measurements,auv.xmeasurements,auv.ymeasurements,gridsize,grid_extent)
            pf,cov = model.evaluate(vals,gno)

            auv.wp = [np.random.randint(0,1000),np.random.randint(0,1000)]
            auv.measurements  = []
            auv.xmeasurements = []
            auv.ymeasurements =[]

            trackx.append(auv.x)
            tracky.append(auv.y)

            fig, ax = plt.subplots(1,2, figsize=(8, 3))
            scp = ax[0].scatter(xval,yval,c=pf)
            ax[0].scatter(auv.x,auv.y,c="red",s=100)
            fig.colorbar(scp,ax=ax[0])
            ax[0].plot(trackx, tracky, c="red")

            scc = ax[1].scatter(xval, yval, c=np.diag(cov),cmap="plasma")
            ax[1].scatter(auv.x, auv.y, c="red", s=100)
            fig.colorbar(scc,ax=ax[1])
            ax[1].plot(trackx, tracky, c="red")
            plt.savefig("fig/2dgp/random" + str(int(figno)) + ".png",dpi = 300)
            plt.show()
            figno += 1


######## Myopic PATH #########
def myopic_evaluate(pf,cov,ax,ay,gridsize,grid_extent,k_mu=1.0, k_cov=1.0):
    radius = 100
    thetas = np.linspace(0, 2 * np.pi,num=50)
    potwpx = radius * np.cos(thetas)
    potwpy = radius * np.sin(thetas)

    c = []
    px = []
    py = []
    for i,x in enumerate(potwpx):
        x = potwpx[i] + ax
        y = potwpy[i] + ay
        if (x < 1000.0) and (x > 0.0) and (y > 0.0) and (y < 1000):
            xx = np.floor(x * gridsize / grid_extent[0])
            yy = np.floor(y * gridsize / grid_extent[1])
            gn = xx * gridsize + yy
            c.append(int(gn))
            px.append(x)
            py.append(y)

    scores = []
    for gg in c:
        scores.append(k_mu*pf[gg]+k_cov*cov[gg,gg]) #point uncertainty

        k_bb = cov[gg,gg]
        k_sb = cov[:, gg]
        k_bs = k_sb.T
        invkb = 1/(k_bb+0.01)
        #scores.append(invkb * np.trace( np.outer(k_sb, k_bs))) # potential uncertainty reduction
        #scores.append(invkb * np.linalg.det(np.outer(k_sb, k_bs)))  # potential uncertainty reduction

    index = np.argmax(scores)
    return px[index],py[index]

auv = vehicle()
model = FlatGaussianRandomField(mu_0, length_scale, gridsize, [1000, 1000], sigma2=0.05, kernel="SE", nugget=0.01)
trackx = [10]
tracky = [10]
figno = 0

mypoic = False
if mypoic:
    for t in range(mission_time):
        auv.update()
        if auv.state:
            vals, gno = segment(auv.measurements, auv.xmeasurements, auv.ymeasurements, gridsize, grid_extent)
            pf, cov = model.evaluate(vals, gno)

            wp = myopic_evaluate(pf,cov,auv.x,auv.y,gridsize,grid_extent,k_mu=0.0)

            auv.wp = wp
            auv.measurements = []
            auv.xmeasurements = []
            auv.ymeasurements = []

            trackx.append(auv.x)
            tracky.append(auv.y)

            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            scp = ax[0].scatter(xval, yval, c=pf)
            ax[0].scatter(auv.x, auv.y, c="red", s=100)
            fig.colorbar(scp, ax=ax[0])
            ax[0].plot(trackx, tracky, c="red")

            scc = ax[1].scatter(xval, yval, c=np.diag(cov), cmap="plasma")
            ax[1].scatter(auv.x, auv.y, c="red", s=100)
            fig.colorbar(scc, ax=ax[1])
            ax[1].plot(trackx, tracky, c="red")
            plt.savefig("fig/2dgp/myopic100mmu0pointuncertainty" + str(int(figno)) + ".png", dpi=300)
            plt.show()
            figno += 1
        print(int(t*100/mission_time), "% done")

######## Non-Myopic PATH #########
def nonmyopic_evaluate(pf,cov,ax,ay,gridsize,grid_extent,k_mu=1.0, k_cov=1.0,search_depth = 3):
    radius = 100
    thetas = np.linspace(0, 2 * np.pi,num=10)
    potwpx = radius * np.cos(thetas)
    potwpy = radius * np.sin(thetas)

    c = []
    px = []
    py = []
    scores = []

    for i,x in enumerate(potwpx):
        x = potwpx[i] + ax
        y = potwpy[i] + ay
        if (x < 1000.0) and (x > 0.0) and (y > 0.0) and (y < 1000):
            xx = np.floor(x * gridsize / grid_extent[0])
            yy = np.floor(y * gridsize / grid_extent[1])
            gn = xx * gridsize + yy
            c.append(int(gn))
            px.append(x)
            py.append(y)

            if search_depth>0:
                scores.append(k_mu*pf[c]+k_cov*cov[c,c]+nonmyopic_evaluate(pf,cov,x,y,gridsize,grid_extent,k_mu=k_mu, k_cov=k_cov,search_depth = search_depth-1)) #point uncertainty

    if search_depth>0:
        return max(scores)
    else:
        index = np.argmax(scores)
        return px[index],py[index]

auv = vehicle()
model = FlatGaussianRandomField(mu_0, length_scale, gridsize, [1000, 1000], sigma2=0.05, kernel="SE", nugget=0.01)
trackx = [10]
tracky = [10]
figno = 0

for t in range(mission_time):
    auv.update()
    if auv.state:
        vals, gno = segment(auv.measurements, auv.xmeasurements, auv.ymeasurements, gridsize, grid_extent)
        pf, cov = model.evaluate(vals, gno)

        wp = nonmyopic_evaluate(pf,cov,auv.x,auv.y,gridsize,grid_extent,k_mu=0.0)

        auv.wp = wp
        auv.measurements = []
        auv.xmeasurements = []
        auv.ymeasurements = []

        trackx.append(auv.x)
        tracky.append(auv.y)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        scp = ax[0].scatter(xval, yval, c=pf)
        ax[0].scatter(auv.x, auv.y, c="red", s=100)
        fig.colorbar(scp, ax=ax[0])
        ax[0].plot(trackx, tracky, c="red")

        scc = ax[1].scatter(xval, yval, c=np.diag(cov), cmap="plasma")
        ax[1].scatter(auv.x, auv.y, c="red", s=100)
        fig.colorbar(scc, ax=ax[1])
        ax[1].plot(trackx, tracky, c="red")
        plt.savefig("fig/2dgp/myopic100mmu0pointuncertainty" + str(int(figno)) + ".png", dpi=300)
        plt.show()
        figno += 1
    print(int(t*100/mission_time), "% done")