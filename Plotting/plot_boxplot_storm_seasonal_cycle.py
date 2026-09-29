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
import matplotlib.dates as dates

###########################################################
###########################################################

wrf_runs = cfg.wrf_runs
lsmooth = 50000 #Smoothing filter length in meters
dx = 2000 #Resolution of imput data
lsmooth_ngrid = np.ceil(lsmooth/dx)
pr_threshold_mmhr = 2.5 #Precipitation threshold in mm/h
freq = '01H'

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

mmh_factor = {'10MIN': 6.,
              '01H'  : 1.,
              'DAY'  : 1/24.}

freq_dt = {'10MIN': 600,
          '01H'  : 3600,
          'DAY'  : 86400.}

pr_threshold = pr_threshold_mmhr/mmh_factor[freq]
dt = freq_dt[freq] #Step between model outputs in seconds
wrun = wrf_runs[0]
reg = 'WME'
syear = 2013
eyear = 2020
smonth = 1
emonth = 12
allmonths = np.arange(1,13)


pr_max = 250
wa_max = 25
xmax = 13
exp = 'exp1'  # reference configuration (ST1)
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

###########################################################
###########################################################

spre = pd.read_pickle(f"storms_pres_summary_{syear}-{eyear}_{allmonths[0]:02d}-{allmonths[-1]:02d}_{reg}_{exp}_{bt_method}_m3.pkl")
sfut = pd.read_pickle(f"storms_fut_summary_{syear}-{eyear}_{allmonths[0]:02d}-{allmonths[-1]:02d}_{reg}_{exp}_{bt_method}_m3.pkl")


spre.start_date = pd.to_datetime(spre['start_date'],format='%Y-%m-%d_%H')
sfut.start_date = pd.to_datetime(sfut['start_date'],format='%Y-%m-%d_%H')
npre = spre.groupby(by=spre.start_date.dt.month).count()
nfut = sfut.groupby(by=sfut.start_date.dt.month).count()
###########################################################
###########################################################
fig = plt.figure(figsize=(10,7))
gs = fig.add_gridspec(2, 1, height_ratios=(2,5),
                      left=0.05, right=0.95, bottom=0.05, top=0.95,
                      wspace=0, hspace=0)
ax = fig.add_subplot(gs[1, 0])
ax_num = fig.add_subplot(gs[0,0], sharex=ax)

bp_pre = spre.boxplot(ax =ax,column=['prmax'],positions=np.arange(0.75,xmax-0.25),by=spre.start_date.dt.month, return_type='dict',patch_artist=True,widths = 0.4)
plt.setp(bp_pre['prmax']['boxes'],color='none')
plt.setp(bp_pre['prmax']['boxes'],facecolor='lightblue')
plt.setp(bp_pre['prmax']['medians'],color='deepskyblue',linewidth=1.)
plt.setp(bp_pre['prmax']['whiskers'], color="deepskyblue", linewidth=0.2,ls='-')
plt.setp(bp_pre['prmax']['fliers'], ms=2,mfc='deepskyblue',markeredgecolor='none', alpha=0.75)
plt.setp(bp_pre['prmax']['caps'], color="deepskyblue", linewidth=0.2)

bp_fut = sfut.boxplot(ax =ax,column=['prmax'],positions=np.arange(1.25,xmax+0.25),by=sfut.start_date.dt.month, return_type='dict',patch_artist=True,widths = 0.4)
plt.setp(bp_fut['prmax']['boxes'],color='none')
plt.setp(bp_fut['prmax']['boxes'],facecolor='lightcoral')
plt.setp(bp_fut['prmax']['medians'],color='crimson',linewidth=1.)
plt.setp(bp_fut['prmax']['whiskers'], color='crimson', linewidth=0.2,ls='-')
plt.setp(bp_fut['prmax']['fliers'],ms=2,mfc='crimson',markeredgecolor='none', alpha=0.75)
plt.setp(bp_fut['prmax']['caps'], color='crimson', linewidth=0.2)

ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('Max PR rate (mm/hr)')
ax.set_ylim([0,pr_max])

ax.set_xlim([0,xmax])
ax.set_xticks(np.arange(len(months))+1)
ax.set_xticklabels(months)

ax.text(0.98,0.9,f"Present",fontsize='medium',fontweight='bold',color="deepskyblue",horizontalalignment='right',transform=ax.transAxes)
ax.text(0.98,0.85,f"Future",fontsize='medium',fontweight='bold',color="crimson",horizontalalignment='right',transform=ax.transAxes)

ax.grid(b=0,axis='x')


ax_num.plot(np.arange(1,xmax),npre.storm_id,linestyle=':',linewidth=0.5,marker='s',mec='none',mfc='deepskyblue',ms=5)
ax_num.plot(np.arange(1,xmax),nfut.storm_id,linestyle='--',linewidth=0.5,marker='s',mec='none',mfc='crimson',ms=5)
ax_num.tick_params(axis="x", labelbottom=False)
ax_num.set_ylim(0,500)
#ax_num.tick_params(axis="y", labelleft=False)

fig.suptitle("Storm Maximum PR Seasonal Cycle", fontsize='20',fontweight='bold')
fig.savefig(f'{cfg.path_figures}/MCS-tracking/Boxplot_storm_seasonalcycle_prmax_{reg}_{exp}.png', bbox_inches='tight',dpi=300)
#plt.savefig("%s/Fig2_%s_Windpower_diurnal_variability_by_%s_WRF_%s-%s.png" %(path_out,wf_name,aggr,syear,eyear),dpi=300)


# #####################################################################
# #####################################################################

# import pdb; pdb.set_trace()

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

# ax.set_ylim(0,wa_max)
# ax.set_xlim(0,pr_max)
# ax.text(0.95,0.95,f"Present",fontsize='xx-large',fontweight='bold',color="deepskyblue",horizontalalignment='right',transform=ax.transAxes)

# plt.setp(ax.get_xticklabels()[-1], visible = False)
# plt.setp(ax.get_yticklabels()[-1], visible = False)
# ax_histx.tick_params(axis="x", labelbottom=False)
# ax_histy.tick_params(axis="y", labelleft=False)

# ax.set_ylabel("Max. W (m/s)", fontsize='xx-large')
# ax.set_xlabel("Max. precip. rate (mm/hr)",fontsize='xx-large')
# sct = ax.scatter(storms_pres_summary.prmax,storms_pres_summary.wa_max,
#            s=(storms_pres_summary.max_size/100)*2,
#            c=storms_pres_summary.tot_vol/1e6,cmap='Spectral',
#            vmin=4, vmax=4000,
#            alpha=0.5)

# ax_histx.hist(storms_pres_summary.prmax,bins = np.linspace(0,pr_max,20),color = "lightgray",alpha=0.8,edgecolor='gray',rwidth=0.8,weights = np.ones_like(storms_pres_summary.prmax) / len(storms_pres_summary.prmax))
# ax_histy.hist(storms_pres_summary.wa_max,bins = np.linspace(0,wa_max,20),color = "lightgray",alpha=0.8,edgecolor='gray', orientation='horizontal',rwidth=0.8,weights = np.ones_like(storms_pres_summary.wa_max) / len(storms_pres_summary.wa_max))

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

# ax2.set_ylim(0,wa_max)
# ax2.set_xlim(0,pr_max)
# ax2.text(0.95,0.95,f"Future",fontsize='xx-large',fontweight='bold',color="crimson",horizontalalignment='right',transform=ax2.transAxes)

# plt.setp(ax2.get_xticklabels()[-1], visible = False)
# plt.setp(ax2.get_yticklabels()[-1], visible = False)
# ax2_histx.tick_params(axis="x", labelbottom=False)
# ax2_histy.tick_params(axis="y", labelleft=False)

# ax2.set_ylabel("Max. W (m/s)", fontsize='xx-large')
# ax2.set_xlabel("Max. precip. rate (mm/hr)", fontsize='xx-large')

# sct2 = ax2.scatter(storms_fut_summary.prmax,storms_fut_summary.wa_max,
#            s=(storms_fut_summary.max_size/100)*2,
#            c=storms_fut_summary.tot_vol/1e6,cmap='Spectral',
#            vmin=4, vmax=4000,
#            alpha=0.5)

# ax2_histx.hist(storms_fut_summary.prmax,bins = np.linspace(0,pr_max,20),color = "lightgray",alpha=0.8,edgecolor='gray',rwidth=0.8,weights = np.ones_like(storms_fut_summary.prmax) / len(storms_fut_summary.prmax))
# ax2_histy.hist(storms_fut_summary.wa_max,bins = np.linspace(0,wa_max,20),color = "lightgray",alpha=0.8,edgecolor='gray', orientation='horizontal',rwidth=0.8,weights = np.ones_like(storms_fut_summary.wa_max) / len(storms_fut_summary.wa_max))

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

# ax3.set_ylim(0,wa_max)
# ax3.set_xlim(0,pr_max)

# plt.setp(ax3.get_xticklabels()[-1], visible = False)
# plt.setp(ax3.get_yticklabels()[-1], visible = False)
# ax3_histx.tick_params(axis="x", labelbottom=False)
# ax3_histy.tick_params(axis="y", labelleft=False)

# ax3.set_ylabel("Max. W (m/s)", fontsize='xx-large')
# ax3.set_xlabel("Max. precip. rate (mm/hr)", fontsize='xx-large')

# sct3p = ax3.scatter(storms_pres_summary.prmax,storms_pres_summary.wa_max,
#            s=(storms_pres_summary.max_size/100)*2,
#            c='deepskyblue',
#            alpha=0.5)

# sct3f = ax3.scatter(storms_fut_summary.prmax,storms_fut_summary.wa_max,
#            s=(storms_fut_summary.max_size/100)*2,
#            c='crimson',
#            alpha=0.5)

# ax3_histx.hist(storms_pres_summary.prmax,bins = np.linspace(0,pr_max,20),color = "deepskyblue",alpha=0.8,edgecolor='deepskyblue',rwidth=0.8, weights = np.ones_like(storms_pres_summary.prmax) / len(storms_pres_summary.prmax))
# ax3_histy.hist(storms_pres_summary.wa_max,bins = np.linspace(0,wa_max,20),color = "deepskyblue",alpha=0.8,edgecolor='deepskyblue', orientation='horizontal',rwidth=0.8,weights = np.ones_like(storms_pres_summary.wa_max) / len(storms_pres_summary.wa_max))

# ax3_histx.hist(storms_fut_summary.prmax,bins = np.linspace(0,pr_max,20),color = "crimson",alpha=0.8,edgecolor='crimson',rwidth=0.8,weights = np.ones_like(storms_fut_summary.prmax) / len(storms_fut_summary.prmax))
# ax3_histy.hist(storms_fut_summary.wa_max,bins = np.linspace(0,wa_max,20),color = "crimson",alpha=0.8,edgecolor='crimson', orientation='horizontal',rwidth=0.8,weights = np.ones_like(storms_fut_summary.wa_max) / len(storms_fut_summary.wa_max))

# ax3.text(0.88,0.9,f"{len(storms_pres_summary)}",fontsize='xx-large',fontweight='bold',color="deepskyblue",transform=ax3.transAxes)
# ax3.text(0.88,0.85,f"{len(storms_fut_summary)}",fontsize='xx-large',fontweight='bold',color="crimson",transform=ax3.transAxes)

# fig.suptitle("Storm characteristics in Western Mediterranean (Aug-Oct)", fontsize='30',fontweight='bold')
# fig.savefig(f'{cfg.path_figures}/MCS-tracking/Scatter_storm_characteristics_WAmax_both_{reg}_{exp}.png', bbox_inches='tight',dpi=300)
