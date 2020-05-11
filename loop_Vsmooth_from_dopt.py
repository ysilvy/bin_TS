import xarray as xr
import numpy as np
import glob
from scipy.optimize import curve_fit
import time
import gsw
import multiprocessing
import pickle

region = 'global'
reso = [0.01,0.02,0.025,0.03,0.04,0.05,0.1]
c = 10
deltaS = reso[1]
deltaT = c*deltaS

# -- Read volume file
V = xr.open_dataset('/data/ysilvy/bin_TS/volumeTS_'+region+'_'+str(deltaS)+'_'+str(deltaT)+'_IPSL-CM5A-LR_historical-rcp85_r2i1p1_1850-2100.nc')
V = V['histogram_so_thetao']

# -- Read dopt file
dopt = xr.open_dataset('/data/ysilvy/bin_TS/dopt_global_0.02_0.2_IPSL-CM5A-LR_historical-rcp85_r2i1p1.nc')
dopt = dopt.doptimal

# -- Select smaller window for testing
V = V.sel(so_bin=slice(33.5,36),thetao_bin=slice(10,18))
dopt = dopt.sel(so_bin=slice(33.5,36),thetao_bin=slice(10,18))

nso = len(V.so_bin.data)
nthetao = len(V.thetao_bin.data)

# -- Compute alpha and beta
S2d,T2d = np.meshgrid(V.so_bin,V.thetao_bin)
P2d = 2000*np.ones(S2d.shape)
alpha = gsw.density.alpha(S2d,T2d,P2d)
beta = gsw.density.beta(S2d,T2d,P2d)

def mean_optwindow(iz):
    # -- Retrieve S and T values from iz index
    S = np.array(V.stack(z=('so_bin','thetao_bin')).isel(z=iz).z.data.max())[0]
    T = np.array(V.stack(z=('so_bin','thetao_bin')).isel(z=iz).z.data.max())[1]
    
    if iz==0:
        print(S,T)
        
    # -- Volume at grid point 
    Vij = V.sel(so_bin=S,thetao_bin=T)
    
#     if Vij.data.all() == 0:
#         V_smooth = xr.zeros_like(Vij)
    
#     else:
        
    # -- Indices of (T,S) grid point
    iS = np.argwhere(V.so_bin.data==Vij.so_bin.data)[0][0]
    iT = np.argwhere(V.thetao_bin.data==Vij.thetao_bin.data)[0][0]
    
    # -- Define distance in T-S space
    adT = 0.5*(alpha + alpha[iT,iS])*abs(T2d-T2d[iT,iS])
    bdS = 0.5*(beta + beta[iT,iS])*abs(S2d-S2d[iT,iS])
    d = np.sqrt(bdS**2 + adT**2) #(T,S)
    
    # -- dopt at grid point
    doptij = dopt.sel(so_bin=S,thetao_bin=T)

    # -- Mean volume for all points where d<=dopt
    V_smooth_dopt = V.where(d.T<=doptij.data).mean(dim=('so_bin','thetao_bin'))
    V_smooth_halfdopt = V.where(d.T<=doptij.data/2).mean(dim=('so_bin','thetao_bin'))
    
    return V_smooth_dopt, V_smooth_halfdopt


print(nso*nthetao, ' steps')
t0=time.time()
with multiprocessing.Pool(4) as p:
    V_smooth_dopt, V_smooth_halfdopt = zip(*p.map(mean_optwindow,np.arange(nso*nthetao)))
#     V_smooth = p.map(mean_optwindow,np.arange(nso*nthetao))
print(time.time() - t0, "seconds wall time")

# -- Turn to DataArray
Vxr1 = xr.DataArray(np.array(V_smooth_dopt).T,dims=V.stack(z=('so_bin','thetao_bin')).dims,coords=V.stack(z=('so_bin','thetao_bin')).coords,name='Vsmooth_dopt')
Vxr1 = Vxr1.unstack()

Vxr2 = xr.DataArray(np.array(V_smooth_halfdopt).T,dims=V.stack(z=('so_bin','thetao_bin')).dims,coords=V.stack(z=('so_bin','thetao_bin')).coords,name='Vsmooth_halfdopt')
Vxr2 = Vxr2.unstack()

Vxr = Vxr1.to_dataset(name=Vxr1.name)
Vxr[Vxr2.name] = Vxr2

# -- Write to out file
Vxr.to_netcdf('/data/ysilvy/bin_TS/volumeTS_smoothdopt_'+region+'_'+str(deltaS)+'_'+str(deltaT)+'_IPSL-CM5A-LR_historical-rcp85_r2i1p1_1850-2100.nc')
