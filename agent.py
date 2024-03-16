import numpy as np
import pandas as pd
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.express as px
import SpatialLogGP
import plotly.graph_objects as go

def getValue(x,y,depth,time=0,noise=True):
    ra = 800.0
    rb = 700.0
    xa = 1500.0
    xb = 2500.0
    ya = 0.0
    yb = 200.0
    za = 17.0
    zb = 32.0
    maxa = 8.0
    maxb = 5.5
    ft = 400.0
    mean_chla = 0.3
    tau = 0.5
    tidalcycle = 44700
    z = depth

    x += 1000*np.sin(2*np.pi*time/tidalcycle)
    y += 800*np.sin(2*np.pi*time/tidalcycle)

    rra = np.sqrt((x-xa)**2 + (y-ya)**2)
    rrb = np.sqrt((x-xb)**2 + (y-yb)**2)
    chla_siga = (np.exp((ra-rra)/ft)/(np.exp((ra-rra)/ft)+1))
    chla_sigb = (np.exp((rb-rrb)/ft)/(np.exp((rb-rrb)/ft)+1))
    chla_sigza = (np.exp(-np.abs(z-za)*1.0)/(np.exp(-np.abs(z-za)*3.0)+1))*2.0
    chla_sigzb = (np.exp(-np.abs(z-zb)*1.0)/(np.exp(-np.abs(z-zb)*3.0)+1))*2.0
    sig_sol = (np.exp(y/ft)/(np.exp(y/ft)+1))
    c_val = mean_chla + z*10.0*np.exp(-z*0.1)*0.01 + 0.1*sig_sol + chla_siga*maxa*chla_sigza + chla_sigb*maxb*chla_sigzb
    if noise:
        return max([c_val + np.random.normal(0.0,c_val*c_val*(np.exp(tau*tau)-1)),0.01])
    else:
        return c_val

class AUV:
    def __init__(self,x_init=-100.0,y_init=-200,update_period = 1,vno=0,no_vs = 4):
        self.x = x_init
        self.y = y_init
        self.z = 0.0

        self.vehicle_no = vno
        self.no_vehicles = no_vs

        self.maxpitch = 20*np.pi/180
        self.speed= 1.5
        self.update_period = update_period

        self.avoidance_coeff = 300

        lat_0 = 78.93295
        lon_0 = 11.953366

        circ = 40075000.0
        self.lat = lat_0 + self.x*360.0/circ
        self.lon = lon_0 + self.y*360.0/(circ*np.cos(lat_0*np.pi/180.0))

        self.state = 0
        self.wp = [0,0,10]

        self.other_vs = []
        for i in range(no_vs):
            self.other_vs.append([-100000, -100000])

        self.grid_extent = [3000, 3000, 50]
        self.grid_size = [15, 15, 10]
        self.lscales = [600,600,3]
        self.sigma2 = 2.3
        self.nugg = 0.7
        self.timesat = 3000
        self.nplaces = self.grid_size[0]*self.grid_size[1]*self.grid_size[2]

        self.divelength = 2 * self.grid_extent[2] / np.tan(self.maxpitch) + 50.0
        self.dives = np.floor(self.grid_extent[0] / self.divelength)

        bl_x = self.divelength / 2.0
        bl_y = (2 * self.vehicle_no + 1) * self.grid_extent[1] / (2.0 * self.no_vehicles)
        self.prior_wps = []
        for i in range(0, int(self.dives * 2) + 1):
            self.prior_wps.append([bl_x*i,bl_y, (i % 2) * self.grid_extent[2] + 0.01])

        self.init = False
        self.adapting = False
        self.prior = False
        self.update_iteration = 0
        self.wp_update_iteration = 0

        self.chlas = []
        self.df = pd.DataFrame(columns=["d", "c", "t"])
        self.df_to_send = pd.DataFrame(columns=["d", "c", "t"])

        self.gp = SpatialLogGP.GriddedLogGaussianProcess3D(self.lscales,
                        self.grid_size, self.grid_extent,self.sigma2,
                        time_sat=self.timesat,nugget=self.nugg)

        self.generatePaths()
        self.xl = []
        self.yl = []
        self.zl = []
        self.uil = []
        self.scorel = []
        self.pmeanl = []
        self.uncl = []

        self.ascending = False
        self.yoyo = True

    def get_xyz(self):
        return self.x, self.y,self.z

    def set_wp(self,wp):
        self.wp = wp
        self.update()

    def update(self):
        self.update_iteration += 1
        ddz = self.grid_extent[2] / self.grid_size[2]
        if self.state:
            self.x = self.x + (self.wp[0]-self.x)*self.speed/np.linalg.norm(np.array([self.x-self.wp[0],self.y-self.wp[1]]))*self.update_period
            self.y = self.y + (self.wp[1]-self.y)*self.speed/np.linalg.norm(np.array([self.x-self.wp[0],self.y-self.wp[1]]))*self.update_period
            #DIVE/ASCENT ENVELOPE WITHIN SAMPLINNG TO USE THE LARGE VERTICAL VARIANCE
            if self.yoyo:
                if self.ascending and abs(self.z-self.wp[2] + ddz) < 0.5:
                    self.ascending = False
                if (not self.ascending) and abs(self.z-self.wp[2] - ddz) < 0.5:
                    self.ascending = True

                if not self.ascending:
                    self.z = self.z + np.sign(self.wp[2]-self.z + ddz)*self.speed*np.sin(self.maxpitch)*self.update_period
                else:
                    self.z = self.z + np.sign(self.wp[2] - self.z - ddz) * self.speed * np.sin(self.maxpitch) * self.update_period

            else:
                if abs(self.wp[2] - self.z) > 0.5:
                    self.z = self.z + np.sign(self.wp[2] - self.z) * self.speed * np.sin(
                        self.maxpitch) * self.update_period

            self.z = max([self.z,0.0])
        self.state = 5 < np.linalg.norm(np.array([self.x-self.wp[0],self.y-self.wp[1]]))

        cv = getValue(self.x,self.y,self.z,time=self.update_iteration*self.update_period)
        self.chlas.append([self.x,self.y,self.z,self.update_iteration*self.update_period,cv])
        self.xl.append(self.x)
        self.yl.append(self.y)
        self.zl.append(-self.z)
        self.uil.append(self.update_iteration)

    def segment(self):
        if len(self.chlas) > 10:
            chlas = np.array(self.chlas)
            x = chlas.T[0]
            y = chlas.T[1]
            z = chlas.T[2]
            t = chlas.T[3]
            data = chlas.T[4]
            xx = np.floor(x * self.grid_size[0] / self.grid_extent[0])
            yy = np.floor(y * self.grid_size[1] / self.grid_extent[1])
            zz = np.floor(z * self.grid_size[2] / self.grid_extent[2])
            gn = xx * self.grid_size[1] * self.grid_size[2] + yy * self.grid_size[2] + zz
            c = gn.astype(int)

            df = pd.DataFrame({"d": data, "c": c, "t": t})
            c = []
            data = []
            t = []
            for i in range(0, self.nplaces):
                d = df[df["c"] == i]
                if len(d) > 0:
                    c.append(i)
                    data.append(d["d"].mean())
                    t.append(d["t"].mean())
            df = pd.DataFrame({"d": data, "c": c, "t": t})
            self.df = pd.concat([self.df, df])
            self.df.sort_values(by="t", inplace=True)
            self.df.drop_duplicates(subset="c", keep="last", inplace=True)
            self.df = self.df[self.df["c"] < self.nplaces]
            self.df = self.df[self.df["c"] >= 0]
            self.chlas = []

    def ingest_df(self,df):
        self.df = pd.concat([self.df, df])
        self.df.sort_values(by="t", inplace=True)
        self.df.drop_duplicates(subset="c", keep="last", inplace=True)

    def generatePaths(self):
        r = np.sqrt(2)*self.grid_extent[0] / self.grid_size[0]
        ddz = self.grid_extent[2]/self.grid_size[2]
        dz = [-ddz,0.0,ddz]
        theta =  np.linspace(0.0, 2.0*np.pi*(1.0-(1.0/8)),8)
        paths = []
        for t in theta:
            for zz in dz:
                paths.append(np.array([r*np.cos(t),r*np.sin(t),zz]))

        self.paths = np.array(paths)

    def adapt(self):
        
        pf, pc = self.gp.evaluate(self.df)
        scores = []
        wps = []
        lmp = []
        lavoid = []
        lunc = []
        pp = copy.deepcopy(self.paths)
        for path in pp:
            path[0] += self.wp[0]
            path[1] += self.wp[1]
            path[2] += self.wp[2]

            if path[2] >= self.grid_extent[2]:
                path[2] =  self.grid_extent[2]-0.01

            if path[2] < 0.0:
                path[2] =  0.01


            if path[0] > 0.0 and path[0] < self.grid_extent[0]:
                if path[1] > 0.0 and path[1] < self.grid_extent[1]:
                    if path[2] > 0.0 and path[2] < self.grid_extent[2]:
                        xx = np.floor(path[0] * self.grid_size[0] / self.grid_extent[0])
                        yy = np.floor(path[1] * self.grid_size[1] / self.grid_extent[1])
                        zz = np.floor(path[2] * self.grid_size[2] / self.grid_extent[2])
                        gn = xx * self.grid_size[1] * self.grid_size[2] + yy * self.grid_size[2] + zz
                        c = gn.astype(int)
                        avoidance_score = 0.0

                        for i,av in enumerate(self.other_vs):
                            if not i == self.vehicle_no:
                                avoidance_score -= self.avoidance_coeff ** 2 / ((path[0] - av[0]) ** 2 + (path[1] - av[1]) ** 2)

                        values = []
                        ss = pc[c, c] #pf[c + i] + pc[c + i, c + i] - avoidance_score
                        values.append(pf[c])
                        lmp.append(pf[c])
                        lunc.append(pc[c,c])
                        lavoid.append(avoidance_score)
                        ii = np.argmax(values)
                        scores.append(ss+pf[c]+avoidance_score)
                        wps.append([path[0], path[1], path[2]])

        index = np.argmax(scores)

        #Log scores, predictive mean, uncertainty, avoidance score, update time, predicted field, and uncertainties
        self.scorel.append([scores[index],lmp[index],lunc[index],lavoid[index],self.update_iteration*self.update_period,pf,np.diag(pc)])
        wp = wps[index]
        return wp

    def ingest_pos(self,pos,vno):
        self.other_vs[vno] = [pos[0],pos[1]]

    def get_state(self):
        return self.state

nv =4
h = AUV(vno=0,no_vs=4)
f = AUV(x_init=200,y_init=100,vno=1,no_vs=4)
r = AUV(x_init=1000,y_init=-500,vno=2,no_vs=4)
t = AUV(x_init=0,y_init=0,vno=3,no_vs=4)
auvs = [h,f,r,t]


run_iterations = 3600*2
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
print(ss.T[(4,)])
print(ss.T[(0,)])

print(len(ss.T[(4,)]))
print(len(ss.T[(0,)]))

plt.plot(ss.T[(4,)],ss.T[(0,)],label = "Total")
#plt.plot(ss.T[(4,)],ss.T[(1,)],label="predictive mean")
#plt.plot(ss.T[(4,)],ss.T[(2,)],label="unc")
#plt.plot(ss.T[(4,)],ss.T[(3,)],label="avoidance")
plt.legend()
plt.savefig("test.png")


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
grid = np.array(grid)
plt.show()
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


