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

def plot_eof_basin(iregion,solver,pseudo_pcs_1,pseudo_pcs_2,pseudo_pcs_3,pseudo_pcs_4,
                  pseudo_pcs_piC1,pseudo_pcs_piC2,pseudo_pcs_piC3,pseudo_pcs_piC4,pseudo_pcs_CO2):

    pcs = solver.pcs()
    eofs = solver.eofs()
    var1 = solver.varianceFraction()[0].data*100
    var2 = solver.varianceFraction()[1].data*100
    var3 = solver.varianceFraction()[2].data*100
    eof1 = eofs[0,:].unstack('z')
    eof2 = -eofs[1,:].unstack('z')
    eof3 = -eofs[2,:].unstack('z')
 
    # -- Plot
    
    stdmax = np.max(np.array([pseudo_pcs_piC1[:,0].std(),pseudo_pcs_piC2[:,0].std(),pseudo_pcs_piC3[:,0].std(),pseudo_pcs_piC4[:,0].std()]))
    baseline = pcs[0:50,0].mean(dim='time')

    time=np.arange(1850,2101)
    timepiC = np.arange(1850,2100)
    timeCO2 = np.arange(1850,1990)
    
    fig,ax = plt.subplots(ncols=2,nrows=3,figsize=(14,15))

    vmin = -0.02 
    vmax = 0.02 

    c = eof1.T.plot(ax=ax[0,0],add_colorbar=False,vmin=vmin,vmax=vmax,center=0,cmap='RdBu_r')
    cb = fig.colorbar(c,ax=ax[0,0],label='')
    ax[0,0].set_title('EOF1 ('+'%.2f '% Decimal(str(var1))+'%)')
    ax[0,1].plot(time,pcs[:,0],label='PC1',lw=2,color='black',zorder=10)
    ax[0,1].plot(time,pseudo_pcs_1[:,0],label='r1i1p1',zorder=9)
    ax[0,1].plot(time,pseudo_pcs_2[:,0],label='r2i1p1',zorder=8)
    ax[0,1].plot(time,pseudo_pcs_3[:,0],label='r3i1p1',zorder=7)
    ax[0,1].plot(time,pseudo_pcs_4[:,0],label='r4i1p1',zorder=6)
    ax[0,1].plot(timeCO2,pseudo_pcs_CO2[:,0],label='1pctCO2')
    ax[0,1].plot(timepiC,pseudo_pcs_piC1[:,0]-abs(baseline),color='grey',label='piControl - ensmean[1850-1900]',zorder=5)
    ax[0,1].plot(timepiC,pseudo_pcs_piC2[:,0]-abs(baseline),color='grey',zorder=4)
    ax[0,1].plot(timepiC,pseudo_pcs_piC3[:,0]-abs(baseline),color='grey',zorder=3)
    ax[0,1].plot(timepiC,pseudo_pcs_piC4[:,0]-abs(baseline),color='grey',zorder=2)
    ax[0,1].fill_between(x=time,y1=baseline-3*stdmax,y2=baseline+3*stdmax,alpha=0.5,color='grey',zorder=0)
    ax[0,1].legend()
    ax[0,1].hlines(0,time[0],time[-1],colors='k',linestyles='dashed')
    ax[0,1].set_title('PC1')
    ylim = ax[0,1].get_ylim()

    c = eof2.T.plot(ax=ax[1,0],add_colorbar=False,vmin=vmin,vmax=vmax,center=0,cmap='RdBu_r')
    cb = fig.colorbar(c,ax=ax[1,0],label='')
    ax[1,0].set_title('EOF2 ('+'%.2f '% Decimal(str(var2))+'%)')
    ax[1,1].plot(time,-pcs[:,1],label='PC2',lw=2,color='black',zorder=10)
    ax[1,1].plot(time,-pseudo_pcs_1[:,1],label='r1i1p1')
    ax[1,1].plot(time,-pseudo_pcs_2[:,1],label='r2i1p1')
    ax[1,1].plot(time,-pseudo_pcs_3[:,1],label='r3i1p1')
    ax[1,1].plot(time,-pseudo_pcs_4[:,1],label='r4i1p1')
    ax[1,1].plot(timeCO2,-pseudo_pcs_CO2[:,1],label='1pctCO2')
    ax[1,1].plot(timepiC,-pseudo_pcs_piC1[:,1],color='grey',label='piControl')
    ax[1,1].plot(timepiC,-pseudo_pcs_piC2[:,1],color='grey')
    ax[1,1].plot(timepiC,-pseudo_pcs_piC3[:,1],color='grey')
    ax[1,1].plot(timepiC,-pseudo_pcs_piC4[:,1],color='grey')
    ax[1,1].hlines(0,time[0],time[-1],colors='k',linestyles='dashed')
    ax[1,1].set_title('PC2')
    ax[1,1].set_ylim(ylim[0],ylim[1])
    ax[1,1].legend()

    c = eof3.T.plot(ax=ax[2,0],add_colorbar=False,vmin=vmin,vmax=vmax,center=0,cmap='RdBu_r')
    cb = fig.colorbar(c,ax=ax[2,0],label='')
    ax[2,0].set_title('EOF3 ('+'%.2f '% Decimal(str(var3))+'%)')
    ax[2,1].plot(time,-pcs[:,2],label='PC3',lw=2,color='black',zorder=10)
    ax[2,1].plot(time,-pseudo_pcs_1[:,2],label='r1i1p1')
    ax[2,1].plot(time,-pseudo_pcs_2[:,2],label='r2i1p1')
    ax[2,1].plot(time,-pseudo_pcs_3[:,2],label='r3i1p1')
    ax[2,1].plot(time,-pseudo_pcs_4[:,2],label='r4i1p1')
    ax[2,1].plot(timeCO2,-pseudo_pcs_CO2[:,2],label='1pctCO2')
    ax[2,1].plot(timepiC,-pseudo_pcs_piC1[:,2],color='grey',label='piControl')
    ax[2,1].plot(timepiC,-pseudo_pcs_piC2[:,2],color='grey')
    ax[2,1].plot(timepiC,-pseudo_pcs_piC3[:,2],color='grey')
    ax[2,1].plot(timepiC,-pseudo_pcs_piC4[:,2],color='grey')
    ax[2,1].hlines(0,time[0],time[-1],colors='k',linestyles='dashed')
    ax[2,1].set_title('PC3')
    ax[2,1].set_ylim(ylim[0],ylim[1])
    ax[2,1].legend()

    plt.subplots_adjust(hspace=0.25,wspace=0.08)
    title = regions[iregion]+'\n historical+rcp85 ensemble mean\nResolution: deltaS='+str(deltaS)+' deltaT='+str(deltaT)
    plt.suptitle(title,fontsize=14,fontweight='bold')
    plt.savefig('eof_ensmean_proj_'+regions[iregion]+'_'+str(deltaS)+'_'+str(deltaT)+'.png')