#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-04-28T13:08:07+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-04-28T13:08:08+02:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import os
import pickle
import numpy as np
import xarray as xr
import netCDF4 as nc
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as cfg
import pandas as pd
import matplotlib.pyplot as plt
import wrf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
import datetime as dt
import matplotlib.gridspec as gridspec

###########################################################
###########################################################

wrf_runs = cfg.wrf_runs
lsmooth = 50000 #Smoothing filter length in meters
dx = 2000 #Resolution of imput data

###########################################################
# Rain volume of a tracked storm
#
# 'tot' in the tracker output is a BARE SUM of the precipitation over the
# cells of the object (mm per time step, summed over cells) - it is not a
# volume. The tracker now also stores 'volume' [m3], computed as
#     V = sum( PR [mm] * cell_area [m2] ) * 1e-3
# i.e. weighting every cell by its own (latitude-dependent) area. Pickles
# produced before that change only have 'tot', so fall back to a uniform-cell
# conversion; on the EPICC grid the real cells are ~4.3 km2 rather than the
# nominal 4.0 km2, so that fallback underestimates the volume by ~7%.
###########################################################
tot_to_m3 = 1e-3 * dx**2   # m3 per summed mm, uniform dx grid

def storm_rain_volume(this_storm, nstep):
    """Rain volume produced by a tracked storm at time step nstep [m3]."""
    if 'volume' in this_storm:
        return this_storm['volume'][nstep]
    return this_storm['tot'][nstep] * tot_to_m3
lsmooth_ngrid = np.ceil(lsmooth/dx)
pr_threshold_mmhr = 2.5 #Precipitation threshold in mm/h
freq = '01H'

mmh_factor = {'10MIN': 6.,
              '01H'  : 1.,
              'DAY'  : 1/24.}

freq_dt = {'10MIN': 600,
          '01H'  : 3600,
          'DAY'  : 86400.}

pr_threshold = pr_threshold_mmhr/mmh_factor[freq]
dt = freq_dt[freq] #Step between model outputs in seconds
reg = 'WME'
syear = 2011
eyear = 2020
smonth = 1
emonth = 12
allmonths = [8,9,10]
calc_summary=False

pr_max = 250
wd_max = 50
exps = ['exp1','exp2','exp3','exp4','exp5']  # exp1 = ST1 (reference)
# Which brightness-temperature conversion the tracking used. 'SB' reads the
# original ConvStormTracking/, 'YS' reads ConvStormTracking_YS/. It also goes
# into the summary filename, so the two sets never overwrite one another.
bt_method = 'SB'
# Both conversions are tracked into ConvStormTracking_<method>. The original
# ConvStormTracking/ holds the first SB run, kept as a record but superseded:
# it predates the area-weighted rain volume, so its volumes are ~7% low.
track_dir = f'ConvStormTracking_{bt_method}'
###########################################################
###########################################################

#Loading geo info from WRF
geofile_ref = nc.Dataset(f'{cfg.geofile_ref}')
xbounds = wrf.cartopy_xlim(wrfin=geofile_ref)
ybounds = wrf.cartopy_ylim(wrfin=geofile_ref)
geo_proj = wrf.get_cartopy(wrfin=geofile_ref)


hgt = wrf.getvar(geofile_ref, "ter")
lats, lons = wrf.latlon_coords(hgt)

period_wrun = {'pres':'EPICC_2km_ERA5','fut':'EPICC_2km_ERA5_CMIP6anom'}


###########################################################
###########################################################
if calc_summary is True:
    for exp in exps:
        for period in ['pres','fut']:
            wrun = period_wrun[period]
            datalist = []
            new_storm_id = 1
            for year in range (syear,eyear+1):
                for month in allmonths:
                    fin = f"{cfg.path_postproc}/{wrun}/{track_dir}/{exp}/MCS_{year}{month:02d}"
                    with open(fin, 'rb') as f:
                        allstorms = pd.read_pickle(f)
                    print(year,month)

                    # WSPD10 is the scalar wind speed averaged over the hour. Deriving it
                    # instead from hourly-mean U and V would give the speed OF the mean
                    # vector, which is always smaller and cancels the rotating, gusty flow
                    # that storms actually have - a bias correlated with storm activity.
                    fin_wspd = xr.open_mfdataset(f"{cfg.path_postproc}/{wrun}/WSPD10/UIB_01H_WSPD10_{year}-{month:02d}.nc")
                    fin_PW = xr.open_mfdataset(f"{cfg.path_postproc}/{wrun}/PW/UIB_03H_PW_{year}-{month:02d}.nc")
                    fin_CAPE = xr.open_mfdataset(f"{cfg.path_postproc}/{wrun}/CAPE2D/UIB_03H_CAPE2D_{year}-{month:02d}.nc")

                    for storm_id in allstorms.keys():
                        this_storm = allstorms[storm_id]
                        print(f'{storm_id}/{len(allstorms)}')
                        for nstep in range(len(this_storm['times'])):

                            if nstep == 0:
                                speed = 0
                            else:
                                speed = this_storm['speed'][nstep-1]

                            wind_area_side = int(np.ceil(np.sqrt(this_storm['size'][nstep]/(dx**2))))


                            x_y = wrf.ll_to_xy(geofile_ref, this_storm['track'][nstep,0],this_storm['track'][nstep,1])

                            x1 = x_y[0].values-wind_area_side
                            x2 = x_y[0].values+wind_area_side
                            y1 = x_y[1].values-wind_area_side
                            y2 = x_y[1].values+wind_area_side

                            if x1<0: x1 = 0
                            if y1<0: y1 = 0
                            if x2>hgt.shape[1]: x2 = hgt.shape[1]
                            if y2>hgt.shape[0]: y2 = hgt.shape[0]



                            # Pair on the CLOCK HOUR, not by nearest stamp. RAIN is stamped at
                            # the middle of its accumulation window (HH:30 in the clock-hour
                            # files, HH:25 in the older ones) while WSPD10, PW and CAPE2D are
                            # stamped on the hour. At HH:30 'nearest' is an exact tie and
                            # pandas rounds UP, silently pairing every storm with the
                            # FOLLOWING hour's fields.
                            hour = pd.Timestamp(this_storm['times'][nstep]).floor('h')
                            wspd_storm = fin_wspd.WSPD10.sel(time=hour).isel(x=slice(x1,x2),y=slice(y1,y2))
                            pw_storm = fin_PW.PW.sel(time=hour-pd.Timedelta(hours=3),method='nearest').isel(x=slice(x1,x2),y=slice(y1,y2))
                            mcape_storm = fin_CAPE.CAPE2D.sel(time=hour-pd.Timedelta(hours=3),method='nearest').isel(x=slice(x1,x2),y=slice(y1,y2),lev=0)
                                #wind_storm_step = fin_wind.sel(time=this_storm['times'][nstep])

                            datalist.append([new_storm_id,
                                            nstep,
                                            this_storm['times'][nstep],
                                            this_storm['size'][nstep],
                                            storm_rain_volume(this_storm, nstep), # rain volume [m3]
                                            this_storm['max'][nstep],
                                            this_storm['mean'][nstep],
                                            speed,
                                            wspd_storm.max().values,
                                            wspd_storm.mean().values,
                                            pw_storm.max().values,
                                            pw_storm.mean().values,
                                            pw_storm.sum().values,
                                            mcape_storm.mean().values,
                                            mcape_storm.max().values,
                                            len(this_storm['times']),
                                            this_storm['track'][nstep,1],
                                            this_storm['track'][nstep,0]
                                            ])
                        new_storm_id+=1
            storms = pd.DataFrame(datalist, columns=['storm_id', 'nstep','datetime',
                                                    'size','prvol','prmax','prmean','speed',
                                                    'wspd_max','wspd_mean',
                                                    'pw_max','pw_mean','pw_sum',
                                                    'mcape_mean','mcape_max',
                                                    'duration','lon','lat'])#,'p80','p85','p90','p95','p99',
                                                    #'hit_end','hit_border'])

            storms = storms.loc[(storms.lon>cfg.reg_coords[reg][1]) & (storms.lon<cfg.reg_coords[reg][3]) & (storms.lat>cfg.reg_coords[reg][0]) & (storms.lat<cfg.reg_coords[reg][2])]
            storms['storm_id']=pd.factorize(storms.storm_id)[0] + 1

            datalist = []
            for nstorm in storms['storm_id'].unique():

                datalist.append([nstorm,
                                storms.loc[storms['storm_id'] == nstorm].prmax.max(),
                                storms.loc[storms['storm_id'] == nstorm].wspd_max.max(),
                                storms.loc[storms['storm_id'] == nstorm]['size'].max()/(1000**2), # max INSTANTANEOUS area over the storm lifetime [km2]
                                storms.loc[storms['storm_id'] == nstorm].prvol.sum(), # rain volume integrated over the lifetime [m3]
                                storms.loc[storms['storm_id'] == nstorm].duration.max(),
                                storms.loc[storms['storm_id'] == nstorm].pw_mean.mean(),
                                storms.loc[storms['storm_id'] == nstorm].pw_sum.sum(),
                                storms.loc[storms['storm_id'] == nstorm].mcape_mean.mean(),
                                storms.loc[storms['storm_id'] == nstorm].mcape_max.max(),
                                ])
            storms_summary = pd.DataFrame(datalist, columns=['storm_id','prmax','wspd_max','max_size','tot_vol','duration','pw_mean','pw_sum','mcape_mean','mcape_max'])
            storms_summary.to_pickle(f"storms_{period}_summary_{syear}-{eyear}_{allmonths[0]:02d}-{allmonths[-1]:02d}_{reg}_{exp}_{bt_method}_m3.pkl")



def _load_summary(tag):
    """Read a storm summary, or explain how to make it."""
    fin = (f"storms_{tag}_summary_{syear}-{eyear}_{allmonths[0]:02d}-"
           f"{allmonths[-1]:02d}_{reg}_{exp}_{bt_method}_m3.pkl")
    if not os.path.exists(fin):
        raise SystemExit(
            f"\nMissing summary: {fin}\n"
            f"  The summaries are built by plot_scatter_hist_storm_characteristics.py\n"
            f"  with calc_summary=True. Set it and run that script once; it reads the\n"
            f"  {bt_method} tracking in {{run}}/{track_dir}/{exp}/ and writes one pickle\n"
            f"  per period. Building takes a while - it opens WSPD10, PW and CAPE2D for\n"
            f"  every month.\n")
    return pd.read_pickle(fin)

    # storms_pres_summary = _load_summary("pres")
    # storms_fut_summary = _load_summary("fut")
###########################################################
###########################################################

# # https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/scatter_hist.html
# fig = plt.figure(figsize=(30, 20))

# gs1 = fig.add_gridspec(2, 2,  width_ratios=(9,9), height_ratios=(9,9),
#                       left=0.05, right=0.95, bottom=0.05, top=0.95,
#                       wspace=0.1, hspace=0.1)

# gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=(7,2), height_ratios=(2,7),subplot_spec=gs1[0,0],wspace=0,hspace=0)
# gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=(7,2), height_ratios=(2,7),subplot_spec=gs1[0,1],wspace=0,hspace=0)
# gs02 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=(7,2), height_ratios=(2,7),subplot_spec=gs1[1,0],wspace=0,hspace=0)
# gs03 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=(7,2), height_ratios=(0.5,8.5),subplot_spec=gs1[1,1],wspace=0,hspace=0)


# ax = fig.add_subplot(gs00[1, 0])
# ax_histx = fig.add_subplot(gs00[0,0], sharex=ax)
# ax_histy = fig.add_subplot(gs00[1,1], sharey=ax)

# ax.set_ylim(0,wd_max)
# ax.set_xlim(0,pr_max)
# ax.text(0.95,0.95,f"Present",fontsize='xx-large',fontweight='bold',color="deepskyblue",horizontalalignment='right',transform=ax.transAxes)
# plt.setp(ax.get_xticklabels()[-1], visible = False)
# plt.setp(ax.get_yticklabels()[-1], visible = False)
# ax_histx.tick_params(axis="x", labelbottom=False)
# ax_histy.tick_params(axis="y", labelleft=False)


# ax.set_ylabel("Max. wind speed (m/s)", fontsize='xx-large')
# ax.set_xlabel("Max. precip. rate (mm/hr)", fontsize='xx-large')


# sct = ax.scatter(storms_pres_summary.prmax,storms_pres_summary.wspd_max,
#            s=(storms_pres_summary.max_size/100)*2,
#            c=storms_pres_summary.tot_vol/1e6,cmap='Spectral',
#            vmin=4, vmax=4000,
#            alpha=0.5)

# ax_histx.hist(storms_pres_summary.prmax,bins = np.linspace(0,pr_max,20),color = "lightgray",alpha=0.8,edgecolor='gray',rwidth=0.8,weights=np.ones_like(storms_pres_summary.prmax) / len(storms_pres_summary.prmax))
# ax_histy.hist(storms_pres_summary.wspd_max,bins = np.linspace(0,wd_max,20),color = "lightgray",alpha=0.8,edgecolor='gray', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_pres_summary.wspd_max) / len(storms_pres_summary.wspd_max))


# kw = dict(prop="sizes", num=5, fmt="{x:.0f}",
#           func=lambda s: s*100/2000)
# legend = ax.legend(*sct.legend_elements(**kw),
#                     loc="lower right", title="Storm size\n ($10^3$ $km^2$)",frameon=False,labelspacing = 2,borderaxespad=2,ncol = 7)
# plt.setp(legend.get_title(), multialignment='center')
# ############################################################
# ###########################################################

# ax2 = fig.add_subplot(gs01[1, 0])
# ax2_histx = fig.add_subplot(gs01[0,0], sharex=ax)
# ax2_histy = fig.add_subplot(gs01[1,1], sharey=ax)

# ax2.set_ylim(0,wd_max)
# ax2.set_xlim(0,pr_max)
# ax2.text(0.95,0.95,f"Future",fontsize='xx-large',fontweight='bold',color="crimson",horizontalalignment='right',transform=ax2.transAxes)

# plt.setp(ax2.get_xticklabels()[-1], visible = False)
# plt.setp(ax2.get_yticklabels()[-1], visible = False)
# ax2_histx.tick_params(axis="x", labelbottom=False)
# ax2_histy.tick_params(axis="y", labelleft=False)


# ax2.set_ylabel("Max. wind speed (m/s)", fontsize='xx-large')
# ax2.set_xlabel("Max. precip. rate (mm/hr)", fontsize='xx-large')




# sct2 = ax2.scatter(storms_fut_summary.prmax,storms_fut_summary.wspd_max,
#            s=(storms_fut_summary.max_size/100)*2,
#            c=storms_fut_summary.tot_vol/1e6,cmap='Spectral',
#            vmin=4, vmax=4000,
#            alpha=0.5)

# ax2_histx.hist(storms_fut_summary.prmax,bins = np.linspace(0,pr_max,20),color = "lightgray",alpha=0.8,edgecolor='gray',rwidth=0.8,weights=np.ones_like(storms_fut_summary.prmax) / len(storms_fut_summary.prmax))
# ax2_histy.hist(storms_fut_summary.wspd_max,bins = np.linspace(0,wd_max,20),color = "lightgray",alpha=0.8,edgecolor='gray', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_fut_summary.wspd_max) / len(storms_fut_summary.wspd_max))

# cbar_ax = fig.add_subplot(gs03[0, 0])
# cbar=plt.colorbar(sct2, cax=cbar_ax,orientation="horizontal")
# cbar.set_label ('Total rain volume ($10^6$ $m^3$)',fontsize='xx-large')
# kw2 = dict(prop="sizes", num=5, fmt="{x:.0f}",
#           func=lambda s: s*100/2000)
# legend2 = ax2.legend(*sct2.legend_elements(**kw2),
#                     loc="lower right", title="Storm size\n ($10^3$ $km^2$)",frameon=False,labelspacing = 2,borderaxespad=2,ncol = 7)
# plt.setp(legend2.get_title(), multialignment='center')

# ###########################################################
# ###########################################################


# ax3 = fig.add_subplot(gs02[1, 0])
# ax3_histx = fig.add_subplot(gs02[0,0], sharex=ax)
# ax3_histy = fig.add_subplot(gs02[1,1], sharey=ax)

# ax3.set_ylim(0,wd_max)
# ax3.set_xlim(0,pr_max)

# ax3_histx.tick_params(axis="x", labelbottom=False)
# ax3_histy.tick_params(axis="y", labelleft=False)


# ax3.set_ylabel("Max. wind speed (m/s)", fontsize='xx-large')
# ax3.set_xlabel("Max. precip. rate (mm/hr)", fontsize='xx-large')


# sct3p = ax3.scatter(storms_pres_summary.prmax,storms_pres_summary.wspd_max,
#            s=(storms_pres_summary.max_size/100)*2,
#            c='deepskyblue',
#            alpha=0.5)

# sct3f = ax3.scatter(storms_fut_summary.prmax,storms_fut_summary.wspd_max,
#            s=(storms_fut_summary.max_size/100)*2,
#            c='crimson',
#            alpha=0.5)

# ax3_histx.hist(storms_pres_summary.prmax,bins = np.linspace(0,pr_max,20),color = "deepskyblue",alpha=0.8,edgecolor='deepskyblue',rwidth=0.8,weights=np.ones_like(storms_pres_summary.prmax) / len(storms_pres_summary.prmax))
# ax3_histy.hist(storms_pres_summary.wspd_max,bins = np.linspace(0,wd_max,20),color = "deepskyblue",alpha=0.8,edgecolor='deepskyblue', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_pres_summary.wspd_max) / len(storms_pres_summary.wspd_max))

# ax3_histx.hist(storms_fut_summary.prmax,bins = np.linspace(0,pr_max,20),color = "crimson",alpha=0.8,edgecolor='crimson',rwidth=0.8,weights=np.ones_like(storms_fut_summary.prmax) / len(storms_fut_summary.prmax))
# ax3_histy.hist(storms_fut_summary.wspd_max,bins = np.linspace(0,wd_max,20),color = "crimson",alpha=0.8,edgecolor='crimson', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_fut_summary.wspd_max) / len(storms_fut_summary.wspd_max))

# ax3.text(0.88,0.9,f"{len(storms_pres_summary)}",fontsize='xx-large',fontweight='bold',color="deepskyblue",transform=ax3.transAxes)
# ax3.text(0.88,0.85,f"{len(storms_fut_summary)}",fontsize='xx-large',fontweight='bold',color="crimson",transform=ax3.transAxes)

# fig.suptitle("Storm characteristics in Western Mediterranean (Aug-Oct)", fontsize='30',fontweight='bold')
# fig.savefig(f'{cfg.path_figures}/MCS-tracking/Scatter_storm_characteristics_both_{reg}_{exp}2.png', bbox_inches='tight',dpi=300)
