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

#Vstack = V.stack(z=('so_bin','thetao_bin'))
#V_smooth = xr.zeros_like(Vstack)
V_ano_end = V.isel(time=slice(-20,-1)).mean(dim='time') - V.sel(time=slice('1850','1900')).mean(dim='time')

nso = len(V.so_bin.data)
nthetao = len(V.thetao_bin.data)

efolding = 1/np.exp(1)

def func(x, b):
    return np.exp(-b * x)

# -- Compute alpha and beta
S2d,T2d = np.meshgrid(V.so_bin,V.thetao_bin)
P2d = 2000*np.ones(S2d.shape)
alpha = gsw.density.alpha(S2d,T2d,P2d)
beta = gsw.density.beta(S2d,T2d,P2d)

# for iz in range(nso*nthetao):
def mean_optwindow(iz):
    # -- Retrieve S and T values from iz index
    S = np.array(V_ano_end.stack(z=('so_bin','thetao_bin')).isel(z=iz).z.data.max())[0]
    T = np.array(V_ano_end.stack(z=('so_bin','thetao_bin')).isel(z=iz).z.data.max())[1]
    
    # -- Volume at grid point 
    Vij = V.sel(so_bin=S,thetao_bin=T)
    
#     if Vij.data.all() == 0:
#         V_smooth = xr.zeros_like(Vij)
#         dopt=0
    
#     else:
        
    # -- Indices of (T,S) gridpoint
    iS = np.argwhere(V.so_bin.data==Vij.so_bin.data)[0][0]
    iT = np.argwhere(V.thetao_bin.data==Vij.thetao_bin.data)[0][0]

    # -- Define distance in T-S space
    adT = 0.5*(alpha + alpha[iT,iS])*abs(T2d-T2d[iT,iS])
    bdS = 0.5*(beta + beta[iT,iS])*abs(S2d-S2d[iT,iS])
    d = np.sqrt(bdS**2 + adT**2) #(T,S)

    # -- Define Vmoy(d) for the gridpoint
    dgrid = np.arange(0,0.0050001,0.00005) #(0,0.0050001,0.00005)
    Vmoy = np.zeros(len(dgrid))
    for i in range(len(dgrid)):
        Vmoy[i] = V_ano_end.where(d.T<=dgrid[i]).mean(dim=('so_bin','thetao_bin'))

    # -- Compute autocorrelation of Vmoy
    autocorr = np.correlate(Vmoy,Vmoy,mode='full')
    autocorr = autocorr/autocorr.max() #Normalize so autocorr[d]=1
    imax=int(len(autocorr)/2) #Keep only middle of the array 

    # -- Fit an explonential decay to autocorr
    popt, pcov = curve_fit(func, dgrid, autocorr[imax:])
    expfit = func(dgrid,*popt)

    # -- Find optimal distance
    idx = np.argmin(abs(expfit-efolding)) #Index where expfit=efolding
    dopt = dgrid[idx] #Optimal distance

    # -- Mean volume for all points where d<=dopt
#     V_smooth = V.where(d.T<=dopt/2).mean(dim=('so_bin','thetao_bin'))
    
    return dopt #, V_smooth


print(nso*nthetao, ' steps')
t0=time.time()
with multiprocessing.Pool(8) as p:
#     dopt, V_smooth = zip(*p.map(mean_optwindow,np.arange(nso*nthetao)))
    dopt = p.map(mean_optwindow,np.arange(nso*nthetao))
print(time.time() - t0, "seconds wall time")

# -- Write to pickle just in case
# pickle.dump( V_smooth, open( "V_smooth3_dopt.pkl", "wb" ) )
pickle.dump(dopt, open( "dopt.pkl", "wb" ) )

# -- Turn to DataArray
# Vxr = xr.DataArray(np.array(V_smooth).T,dims=V.stack(z=('so_bin','thetao_bin')).dims,coords=V.stack(z=('so_bin','thetao_bin')).coords,name='Vsmooth')
# Vxr = Vxr.unstack()

dopt_xr = xr.DataArray(np.array(dopt),dims=V_ano_end.stack(z=('so_bin','thetao_bin')).dims,coords=V_ano_end.stack(z=('so_bin','thetao_bin')).coords,name='doptimal')
dopt_xr = dopt_xr.unstack()

# -- Write to out file
# Vxr.to_netcdf('/data/ysilvy/bin_TS/volumeTS_meandopt3_'+region+'_'+str(deltaS)+'_'+str(deltaT)+'_IPSL-CM5A-LR_historical-rcp85_r2i1p1_1850-2100.nc')

dopt_xr.to_netcdf('/data/ysilvy/bin_TS/dopt_'+region+'_'+str(deltaS)+'_'+str(deltaT)+'_IPSL-CM5A-LR_historical-rcp85_r2i1p1.nc')