""" Volume change in TS diagram between end of historical and mid 20th century """

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xhistogram.xarray import histogram

# Choose files
datadirT = '/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Odec/thetao/gn/latest/'
fileT = 'thetao_Odec_IPSL-CM6A-LR_historical_r1i1p1f1_gn_1855-2005.nc'
datadirS = '/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Odec/so/gn/latest/'
fileS = 'so_Odec_IPSL-CM6A-LR_historical_r1i1p1f1_gn_1855-2005.nc'
datadirareacello = '/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Ofx/areacello/gn/latest/'
fileareacello = 'areacello_Ofx_IPSL-CM6A-LR_historical_r1i1p1f1_gn.nc'

# Read files
Tds = xr.open_dataset(datadirT+fileT)
Sds = xr.open_dataset(datadirS+fileS)
a = xr.open_dataset(datadirareacello + fileareacello)

# Create volume per grid cell
dz = np.diff(Tds.olevel)
dz = np.insert(dz, 0, dz[0])
dz = xr.DataArray(dz, coords= {'olevel':Tds.olevel}, dims='olevel')
dVol = a.areacello*dz

# Define bin sizes
sbins = np.arange(31,38, 0.025)
tbins = np.arange(-2, 32, 0.1)

# Select period and read T/S data
T2 = Tds.thetao[-2:,:,:,:]
S2 = Sds.so
T1 = Tds.thetao[0:2,:,:,:]
S1 = Sds.so[0:2,:,:,:]

T1clim = T1.mean(dim='time')
S1clim = S1.mean(dim='time')
T2clim = T2.mean(dim='time')
S2clim = S2.mean(dim='time')

# Make histograms
hTS1 = histogram(S1clim, T1clim, bins=[sbins,tbins], weights=dVol)
hTS2 = histogram(S2clim, T2clim, bins=[sbins,tbins], weights=dVol)

hTS_change = hTS2-hTS1

(hTS_change.T).plot(vmin=-5.e14, vmax=5.e14, cmap='bwr')
