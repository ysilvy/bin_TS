import xarray as xr
import numpy as np
from eofs.xarray import Eof
import matplotlib.pyplot as plt

def compute_eof(V):
    # -- Center
    V_red = V - V.mean(dim='time')
    # -- Reduce
    V_red = V_red/V.std(dim='time')
    # -- Replace nans with zeros
    V_red = xr.where(np.isnan(V_red)==False,V_red,0)
    # -- Load into memory
    V_red = V_red.compute()
    # -- EOF
    solver=Eof(V_red.stack(z=('so_bin','thetao_bin')))
    return solver

def proj_on_eof(V,solver):
    # -- Center
    V_red = V - V.mean(dim='time')
    # -- Reduce
    V_red = V_red/V.std(dim='time')
    # -- Replace nans with zeros
    V_red = xr.where(np.isnan(V_red)==False,V_red,0)
    V_red = V_red.compute()
    # -- Projection
    pseudo_pcs = solver.projectField(V_red.stack(z=('so_bin', 'thetao_bin')))
    return pseudo_pcs

