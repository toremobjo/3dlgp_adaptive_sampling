import numpy as np


def getValue(x,y,depth,time=0,noise=True):
    ra = 800.0
    rb = 700.0
    xa = 1500.0
    xb = 2500.0
    ya = 1800.0
    yb = 200.0
    za = 17.0
    zb = 32.0
    maxa = 8.0
    maxb = 5.5
    ft = 800.0
    mean_chla = 0.3
    tau = 0.4
    tidalcycle = 44700
    z = depth

    x += 1000*np.sin(2*np.pi*time/tidalcycle)
    y += 800*np.sin(2*np.pi*time/tidalcycle)
    z += 1.5 * np.sin(2 * np.pi * time / tidalcycle)

    rra = np.sqrt((x-xa)**2 + (y-ya)**2)
    rrb = np.sqrt((x-xb)**2 + (y-yb)**2)
    chla_siga = (np.exp((ra-rra)/ft)/(np.exp((ra-rra)/ft)+1))
    chla_sigb = (np.exp((rb-rrb)/ft)/(np.exp((rb-rrb)/ft)+1))
    chla_sigza = (np.exp(-np.abs(z-za)*1.0)/(np.exp(-np.abs(z-za)*2.5)+1))*2.0
    chla_sigzb = (np.exp(-np.abs(z-zb)*0.2)/(np.exp(-np.abs(z-zb)*0.5)+1))*2.0
    sig_sol = (np.exp(y/ft)/(np.exp(y/ft)+1))
    c_val = mean_chla + z*10.0*np.exp(-z*0.1)*0.01 + 0.1*sig_sol + chla_siga*maxa*chla_sigza + chla_sigb*maxb*chla_sigzb
    if noise:
        return max([c_val + np.random.normal(0.0,c_val*c_val*(np.exp(tau*tau)-1)),0.01])
    else:
        return c_val
