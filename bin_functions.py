import xarray as xr
import numpy as np
from xhistogram.xarray import histogram

def bin_TS(S,T,vol,sbins,tbins):
    """Bin volume at each time step and create 3rd dimension to save data
    S: Salinity xarray dataset 4D time, lon, lat, lev
    T: Temperature xarray dataset 4D
    vol: Volume xarray dataarray (volume at each cell grid) 3D
    sbins: Salinity bin edges, 1D
    tbins: Temperature bin edges, 1D
    
    V: Volume xarray dataset (time, so_bin, thetao_bin) so_bin and thetao_bin are the middle of the bins
    """
    # Initialize
    hTS_0 = histogram(S.so.isel(time=0), T.thetao.isel(time=0)-273.15, bins=[sbins,tbins], weights=vol)
    hTS_0['time'] = T.time[0]
    hTS_0 = hTS_0.expand_dims('time') 
    V = hTS_0
    # Loop
    for it in range(1,len(T.time)):
        hTS = histogram(S.so.isel(time=it), T.thetao.isel(time=it)-273.15, bins=[sbins,tbins], weights=vol)
        hTS['time'] = T.time[it]
        hTS = hTS.expand_dims('time') 
        V = xr.concat([V, hTS], dim='time')
    return V

def select_basin(ds,mask):
    ds_out = ds.copy(deep=True)
    # Southern Ocean
    ds_out = ds.where(mask.nav_lat<=-35).where(ds<1.e19)
    ds_out['region']='southern_ocean'
    # Atlantic with Arctic
    ds_atl = ds.where(mask.atlmsk==1).where(mask.nav_lat>-35); ds_atl = xr.where(mask.arpmsk==1,ds,ds_atl)
    ds_atl['region']='atlantic'
    ds_out = xr.concat([ds_out,ds_atl], dim='region')
    # Pacific no Arctic
    ds_pac = xr.where(mask.pacmsk==1,ds,np.nan).where(mask.nav_lat>-35).where(mask.arpmsk==0)
    ds_pac = ds_pac.assign_coords(region='pacific').expand_dims('region')
    ds_out = xr.concat([ds_out,ds_pac], dim='region')
    # Indian
    ds_ind = xr.where(mask.indmsk==1,ds,np.nan).where(mask.nav_lat>-35)
    ds_ind = ds_ind.assign_coords(region='indian').expand_dims('region')
    ds_out = xr.concat([ds_out,ds_ind], dim='region')
    ds_out = ds_out.drop('time_counter')
    return ds_out

def select_basin_EN4(dsEN4,maskEN4):
    ds_out = dsEN4.copy(deep=True)
    # Southern Ocean
    ds_out = dsEN4.where(dsEN4.lat<=-35)
    ds_out['region']='southern_ocean'
    # Atlantic
    ds_atl = dsEN4.where(maskEN4==1).where(dsEN4.lat>-35)
    ds_atl['region']='atlantic'
    ds_out = xr.concat([ds_out,ds_atl], dim='region')
    # Pacific
    ds_pac = dsEN4.where(maskEN4==2).where(dsEN4.lat>-35)
    ds_pac = ds_pac.assign_coords(region='pacific').expand_dims('region')
    ds_out = xr.concat([ds_out,ds_pac], dim='region')
    # Indian
    ds_ind = dsEN4.where(maskEN4==3).where(dsEN4.lat>-35)
    ds_ind = ds_ind.assign_coords(region='indian').expand_dims('region')
    ds_out = xr.concat([ds_out,ds_ind], dim='region')
    ds_out = ds_out.drop(('time_bnds','depth_bnds','lon_bnds','lat_bnds'))
    return ds_out

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)