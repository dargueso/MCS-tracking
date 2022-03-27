#!/usr/bin/env python

'''
   Tracking_Functions.py

   This file contains the tracking fuctions for the object
   identification and tracking of precipitation areas, cyclones,
   clouds, and moisture streams

'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from pdb import set_trace as stop
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle
import datetime
import pandas as pd
import subprocess
import matplotlib.path as mplPath
import sys
from calendar import monthrange
from itertools import groupby
from tqdm import tqdm
import time

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import h5py
import xarray as xr
import netCDF4


# ==============================================================
# ==============================================================

# def ObjectCharacteristics(PR_objectsFull, # feature object file
#                          PR_orig,         # original file used for feature detection
#                          SaveFile,        # output file name and locaiton
#                          TIME,            # timesteps of the data
#                          Lat,             # 2D latidudes
#                          Lon,             # 2D Longitudes
#                          Gridspacing,     # average grid spacing
#                          MinTime=1,       # minimum lifetime of an object
#                          Boundary = 1):   # 1 --> remove object when it hits the boundary of the domain

#     import scipy
#     import pickle

#     nr_objectsUD=PR_objectsFull.max()
#     rgiObjectsUDFull = PR_objectsFull
#     if nr_objectsUD >= 1:
#         grObject={}
#         print('            Loop over '+str(PR_objectsFull.max())+' objects')
#         for ob in range(int(PR_objectsFull.max())):
# #             print('        process object '+str(ob+1)+' out of '+str(PR_objectsFull.max()))
#             TT=(np.sum((PR_objectsFull == (ob+1)), axis=(1,2)) > 0)
#             if sum(TT) >= MinTime:
#                 PR_objects=PR_objectsFull[TT,:,:]
#                 rgrObAct=np.array(np.copy(PR_orig[TT,:,:])).astype('float')
#                 rgrObAct[PR_objects != (ob+1)]=0
#                 rgiObjectsUD=rgiObjectsUDFull[TT,:,:]
#                 TimeAct=TIME[TT]

#                 # Does the object hit the boundary?
#                 rgiObjActSel=np.array(PR_objects == (ob+1)).astype('float')
#                 if Boundary == 1:
#                     rgiBoundary=(np.sum(rgiObjActSel[:,0,:], axis=1)+np.sum(rgiObjActSel[:,-1,:], axis=1)+np.sum(rgiObjActSel[:,:,0], axis=1)+np.sum(rgiObjActSel[:,:,-1], axis=1) != 0)
#                     rgrObAct[rgiBoundary,:,:]=np.nan
#                     rgiObjActSel[rgiBoundary,:,:]=np.nan
#                 rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(rgrObAct[tt,:,:]) for tt in range(PR_objects.shape[0])])
#                 rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(PR_objects.shape[0]-1)])*(Gridspacing/1000.)

#                 # plt.plot(rgrObjSpeed); plt.plot(SpeedRMSE); plt.plot(SpeedCorr); plt.plot(SpeedAverage, c='k', lw=3); plt.show()
#                 SpeedAverage=np.copy(rgrObjSpeed) #np.nanmean([rgrObjSpeed,SpeedRMSE,SpeedCorr], axis=0)
#                 rgrPR_Vol=(np.array([np.sum(rgrObAct[tt,:,:]) for tt in range(PR_objects.shape[0])])/(12.*60.*5.))*Gridspacing**2
#                 rgrPR_Max=np.array([np.max(rgrObAct[tt,:,:]) for tt in range(PR_objects.shape[0])])
# #                 rgrPR_Percentiles = np.zeros((rgiObjectsUD.shape[0],101)); rgrPR_Percentiles[:] = np.nan
#                 rgrPR_Mean = np.zeros((rgiObjectsUD.shape[0])); rgrPR_Mean[:] = np.nan
#                 for tt in range(rgiObjectsUD.shape[0]):
#                     if np.sum(rgiObjectsUD[tt,:,:] == (ob+1)) >0:
# #                         PR_perc=np.percentile(rgrObAct[tt,:,:][rgiObjectsUD[tt,:,:] == (ob+1)], range(101))
#                         PR_mean=np.mean(rgrObAct[tt,:,:][rgiObjectsUD[tt,:,:] == (ob+1)])
#                     else:
#                         PR_mean=np.nan
# #                         PR_perc=np.array([np.nan]*101)
# #                     rgrPR_Percentiles[tt,:] = PR_perc
#                     rgrPR_Mean[tt] = PR_mean

#                 rgrSize=np.array([np.sum(rgiObjActSel[tt,:,:] == 1) for tt in range(rgiObjectsUD.shape[0])])*(Gridspacing/1000.)**2
#                 rgrSize[(rgrSize == 0)]=np.nan

#                 # Track lat/lon
#                 TrackAll = np.zeros((len(rgrMassCent),2)); TrackAll[:] = np.nan
#                 try:
#                     FIN = ~np.isnan(rgrMassCent[:,0])
#                     for ii in range(len(rgrMassCent)):
#                         if ~np.isnan(rgrMassCent[ii,0]) == True:
#                             TrackAll[ii,1] = Lat[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
#                             TrackAll[ii,0] = Lon[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
#                 except:
#                     stop()

#                 grAct={'rgrMassCent':rgrMassCent,
#                        'rgrObjSpeed':SpeedAverage,
#                        'rgrPR_Vol':rgrPR_Vol,
# #                        'rgrPR_Percentiles':rgrPR_Percentiles,
#                        'rgrPR_Max':rgrPR_Max,
#                        'rgrPR_Mean':rgrPR_Mean,
#                        'rgrSize':rgrSize,
# #                        'rgrAccumulation':rgrAccumulation,
#                        'TimeAct':TimeAct,
#                        'rgrMassCentLatLon':TrackAll}
#                 try:
#                     grObject[str(ob+1)]=grAct
#                 except:
#                     stop()
#                     continue
#         if SaveFile != None:
#             pickle.dump(grObject, open(SaveFile, "wb" ) )
#         return grObject

def ObjectCharacteristics(PR_objectsFull, # feature object file
                         PR_orig,         # original file used for feature detection
                         SaveFile,        # output file name and locaiton
                         TIME,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,     # average grid spacing
                         Area,
                         MinTime=1,       # minimum lifetime of an object
                         Boundary = 1):   # 1 --> remove object when it hits the boundary of the domain


    # ========

    import scipy
    import pickle

    nr_objectsUD=PR_objectsFull.max()
    rgiObjectsUDFull = PR_objectsFull
    if nr_objectsUD >= 1:
        grObject={}
        print('            Loop over '+str(PR_objectsFull.max())+' objects')
        for ob in range(int(PR_objectsFull.max())):
    #             print('        process object '+str(ob+1)+' out of '+str(PR_objectsFull.max()))
            TT=(np.sum((PR_objectsFull == (ob+1)), axis=(1,2)) > 0)
            if sum(TT) >= MinTime:
                PR_object=np.copy(PR_objectsFull[TT,:,:])
                PR_object[PR_object != (ob+1)]=0
                Objects=ndimage.find_objects(PR_object)
                if len(Objects) > 1:
                    Objects = [Objects[np.where(np.array(Objects) != None)[0][0]]]

                ObjAct = PR_object[Objects[0]]
                ValAct = PR_orig[TT,:,:][Objects[0]]
                ValAct[ObjAct == 0] = np.nan
                AreaAct = np.repeat(Area[Objects[0][1:]][None,:,:], ValAct.shape[0], axis=0)
                AreaAct[ObjAct == 0] = np.nan
                LatAct = np.copy(Lat[Objects[0][1:]])
                LonAct = np.copy(Lon[Objects[0][1:]])

                # calculate statistics
                TimeAct=TIME[TT]
                rgrSize = np.nansum(AreaAct, axis=(1,2))
                rgrPR_Min = np.nanmin(ValAct, axis=(1,2))
                rgrPR_Max = np.nanmax(ValAct, axis=(1,2))
                rgrPR_Mean = np.nanmean(ValAct, axis=(1,2))
                rgrPR_Vol = np.nansum(ValAct, axis=(1,2))

                # Track lat/lon
                rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(ObjAct[tt,:,:]) for tt in range(ObjAct.shape[0])])
                TrackAll = np.zeros((len(rgrMassCent),2)); TrackAll[:] = np.nan
                try:
                    FIN = ~np.isnan(rgrMassCent[:,0])
                    for ii in range(len(rgrMassCent)):
                        if ~np.isnan(rgrMassCent[ii,0]) == True:
                            TrackAll[ii,1] = LatAct[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                            TrackAll[ii,0] = LonAct[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                except:
                    stop()

                rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(ValAct.shape[0]-1)])*(Gridspacing/1000.)

                grAct={'rgrMassCent':rgrMassCent,
                       'rgrObjSpeed':rgrObjSpeed,
                       'rgrPR_Vol':rgrPR_Vol,
                       'rgrPR_Min':rgrPR_Min,
                       'rgrPR_Max':rgrPR_Max,
                       'rgrPR_Mean':rgrPR_Mean,
                       'rgrSize':rgrSize,
    #                        'rgrAccumulation':rgrAccumulation,
                       'TimeAct':TimeAct,
                       'rgrMassCentLatLon':TrackAll}
                try:
                    grObject[str(ob+1)]=grAct
                except:
                    stop()
                    continue
        if SaveFile != None:
            pickle.dump(grObject, open(SaveFile, "wb" ) )
        return grObject


# ==============================================================
# ==============================================================

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import h5py
import xarray as xr

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

# ==============================================================
# ==============================================================

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


# ==============================================================
# ==============================================================
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)


# ==============================================================
# ==============================================================
def Feature_Calculation(DATA_all,    # np array that contains [time,lat,lon,Variables] with vars
                        Variables,   # Variables beeing ['V', 'U', 'T', 'Q', 'SLP']
                        dLon,        # distance between longitude cells
                        dLat,        # distance between latitude cells
                        Lat,         # Latitude coordinates
                        dT,          # time step in hours
                        Gridspacing):# grid spacing in m
    from scipy import ndimage


    # 11111111111111111111111111111111111111111111111111
    # calculate vapor transport on pressure level
    VapTrans = ((DATA_all[:,:,:,Variables.index('U')]*DATA_all[:,:,:,Variables.index('Q')])**2 + (DATA_all[:,:,:,Variables.index('V')]*DATA_all[:,:,:,Variables.index('Q')])**2)**(1/2)

    # 22222222222222222222222222222222222222222222222222
    # Frontal Detection according to https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073662
    UU = DATA_all[:,:,:,Variables.index('U')]
    VV = DATA_all[:,:,:,Variables.index('V')]
    dx = dLon
    dy = dLat
    du = np.gradient( UU )
    dv = np.gradient( VV )
    PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
    TK = DATA_all[:,:,:,Variables.index('T')]
    vgrad = np.gradient(TK, axis=(1,2))
    Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

    Fstar = PV * Tgrad

    Tgrad_zero = 0.45#*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
    import metpy.calc as calc
    from metpy.units import units
    CoriolisPar = calc.coriolis_parameter(np.deg2rad(Lat))
    Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))

    # # 3333333333333333333333333333333333333333333333333333
    # # Cyclone identification based on pressure annomaly threshold

    SLP = DATA_all[:,:,:,Variables.index('SLP')]/100.
    # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
    SLP_smooth = ndimage.uniform_filter(SLP, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
    # smoothign over 3000 x 3000 km and 78 hours
    SLPsmoothAn = ndimage.uniform_filter(SLP, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    SLP_Anomaly = np.array(SLP_smooth-SLPsmoothAn)
    # plt.contour(SLP_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = SLP_Anomaly < -12 # 12 hPa depression
    HighPressure_annomaly = SLP_Anomaly > 12

    return Pressure_anomaly, Frontal_Diagnostic, VapTrans, SLP_Anomaly, vgrad, HighPressure_annomaly



# ==============================================================
# ==============================================================
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km



def ReadERA5(TIME,      # Time period to read (this program will read hourly data)
            var,        # Variable name. See list below for defined variables
            PL,         # Pressure level of variable
            REGION):    # Region to read. Format must be <[N,E,S,W]> in degrees from -180 to +180 longitude
    # ----------
    # This function reads hourly ERA5 data for one variable from NCAR's RDA archive in a region of interest.
    # ----------

    DayStart = datetime.datetime(TIME[0].year, TIME[0].month, TIME[0].day,TIME[0].hour)
    DayStop = datetime.datetime(TIME[-1].year, TIME[-1].month, TIME[-1].day,TIME[-1].hour)
    TimeDD=pd.date_range(DayStart, end=DayStop, freq='d')
    Plevels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])

    dT = int(divmod((TimeDD[1] - TimeDD[0]).total_seconds(), 60)[0]/60)

    # check if variable is defined
    if var == 'V':
        ERAvarfile = 'v.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'V'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'U':
        ERAvarfile = 'u.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'U'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'T':
        ERAvarfile = 't.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'T'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'ZG':
        ERAvarfile = 'z.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Z'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'Q':
        ERAvarfile = 'q.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Q'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'SLP':
        ERAvarfile = 'msl.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.sfc/'
        NCvarname = 'MSL'
        PL = -1
    if var == 'IVTE':
        ERAvarfile = 'viwve.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVE'
        PL = -1
    if var == 'IVTN':
        ERAvarfile = 'viwvn.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVN'
        PL = -1

    print(ERAvarfile)
    # read in the coordinates
    ncid=Dataset("/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", mode='r')
    Lat=np.squeeze(ncid.variables['latitude'][:])
    Lon=np.squeeze(ncid.variables['longitude'][:])
    # Zfull=np.squeeze(ncid.variables['Z'][:])
    ncid.close()
    if np.max(Lon) > 180:
        Lon[Lon >= 180] = Lon[Lon >= 180] - 360
    Lon,Lat = np.meshgrid(Lon,Lat)

    # get the region of interest
    if (REGION[1] > 0) & (REGION[3] < 0):
        # region crosses zero meridian
        iRoll = np.sum(Lon[0,:] < 0)
    else:
        iRoll=0
    Lon = np.roll(Lon,iRoll, axis=1)
    iNorth = np.argmin(np.abs(Lat[:,0] - REGION[0]))
    iSouth = np.argmin(np.abs(Lat[:,0] - REGION[2]))+1
    iEeast = np.argmin(np.abs(Lon[0,:] - REGION[1]))+1
    iWest = np.argmin(np.abs(Lon[0,:] - REGION[3]))
    print(iNorth,iSouth,iWest,iEeast)

    Lon = Lon[iNorth:iSouth,iWest:iEeast]
    Lat = Lat[iNorth:iSouth,iWest:iEeast]
    # Z=np.roll(Zfull,iRoll, axis=1)
    # Z = Z[iNorth:iSouth,iWest:iEeast]

    DataAll = np.zeros((len(TIME),Lon.shape[0],Lon.shape[1])); DataAll[:]=np.nan
    tt=0

    for mm in range(len(TimeDD)):
        YYYYMM = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)
        YYYYMMDD = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)+str(TimeDD[mm].day).zfill(2)
        DirAct = Dir + YYYYMM + '/'
        if (var == 'SLP') | (var == 'IVTE') | (var == 'IVTN'):
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMM+'*.nc')
        else:
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMMDD+'*.nc')
        FILES = np.sort(FILES)

        TIMEACT = TIME[(TimeDD[mm].year == TIME.year) &  (TimeDD[mm].month == TIME.month) & (TimeDD[mm].day == TIME.day)]

        for fi in range(len(FILES)): #[7:9]:
            print(FILES[fi])
            ncid = Dataset(FILES[fi], mode='r')
            time_var = ncid.variables['time']
            dtime = netCDF4.num2date(time_var[:],time_var.units)
            TimeNC = pd.to_datetime([pd.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dtime])
            TT = np.isin(TimeNC, TIMEACT)
            if iRoll != 0:
                if PL !=-1:
                    try:
                        DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,:])
                    except:
                        stop()
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,:])
                ncid.close()
            else:
                if PL !=-1:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,iWest:iEeast])
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,iWest:iEeast])
                ncid.close()
            # cut out region
            if len(DATAact.shape) == 2:
                DATAact=DATAact[None,:,:]
            DATAact=np.roll(DATAact,iRoll, axis=2)
            if iRoll != 0:
                DATAact = DATAact[:,:,iWest:iEeast]
            else:
                DATAact = DATAact[:,:,:]
            try:
                DataAll[tt:tt+DATAact.shape[0],:,:]=DATAact
            except:
                continue
            tt = tt+DATAact.shape[0]
    return DataAll, Lat, Lon


def ConnectLon(Objects):
    for tt in range(Objects.shape[0]):
        EDGE = np.append(Objects[tt,:,-1][:,None],Objects[tt,:,0][:,None], axis=1)
        iEDGE = (np.sum(EDGE>0, axis=1) == 2)
        OBJ_Left = EDGE[iEDGE,0]
        OBJ_Right = EDGE[iEDGE,1]
        OBJ_joint = np.array([OBJ_Left[ii].astype(str)+'_'+OBJ_Right[ii].astype(str) for ii in range(len(OBJ_Left))])
        NotSame = OBJ_Left != OBJ_Right
        OBJ_joint = OBJ_joint[NotSame]
        OBJ_unique = np.unique(OBJ_joint)
        # set the eastern object to the number of the western object in all timesteps
        for ob in range(len(OBJ_unique)):
            ObE = int(OBJ_unique[ob].split('_')[1])
            ObW = int(OBJ_unique[ob].split('_')[0])
            Objects[Objects == ObE] = ObW
    return Objects


def ConnectLonOld(rgiObjectsAR):
    # connect objects allong date line
    for tt in range(rgiObjectsAR.shape[0]):
        for y in range(rgiObjectsAR.shape[1]):
            if rgiObjectsAR[tt, y, 0] > 0 and rgiObjectsAR[tt, y, -1] > 0:
#                 rgiObjectsAR[rgiObjectsAR == rgiObjectsAR[tt, y, -1]] = rgiObjectsAR[tt, y, 0]
                COPY_Obj_tt = np.copy(rgiObjectsAR[tt,:,:])
                COPY_Obj_tt[COPY_Obj_tt == rgiObjectsAR[tt, y, -1]] = rgiObjectsAR[tt, y, 0]
                rgiObjectsAR[tt,:,:] = COPY_Obj_tt
    return(rgiObjectsAR)


# In[228]:


### Break up long living cyclones by extracting the biggest cyclone at each time
def BreakupObjects(DATA,     # 3D matrix [time,lat,lon] containing the objects
                  MinTime,    # minimum volume of each object
                  dT):       # time step in hours

    Objects = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(24/dT) # min livetime of object to be split
    AVmax = 1.5

    rgiObj_Struct2D = np.zeros((3,3,3)); rgiObj_Struct2D[1,:,:]=1
    rgiObjects2D, nr_objects2D = ndimage.label(DATA, structure=rgiObj_Struct2D)

    rgiObjNrs = np.unique(DATA)[1:]
    TT = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(MaxOb)])
    # Sel_Obj = rgiObjNrs[TT > MinLif]


    # Average 2D objects in 3D objects?
    Av_2Dob = np.zeros((len(rgiObjNrs))); Av_2Dob[:] = np.nan
    ii = 1
    for ob in range(len(rgiObjNrs)):
#         if TT[ob] <= MinLif:
#             # ignore short lived objects
#             continue
        SelOb = rgiObjNrs[ob]-1
        DATA_ACT = np.copy(DATA[Objects[SelOb]])
        iOb = rgiObjNrs[ob]
        rgiObjects2D_ACT = np.copy(rgiObjects2D[Objects[SelOb]])
        rgiObjects2D_ACT[DATA_ACT != iOb] = 0

        Av_2Dob[ob] = np.mean(np.array([len(np.unique(rgiObjects2D_ACT[tt,:,:]))-1 for tt in range(DATA_ACT.shape[0])]))
        if Av_2Dob[ob] > AVmax:
            ObjectArray_ACT = np.copy(DATA_ACT); ObjectArray_ACT[:] = 0
            rgiObAct = np.unique(rgiObjects2D_ACT[0,:,:])[1:]
            for tt in range(1,rgiObjects2D_ACT[:,:,:].shape[0]):
                rgiObActCP = list(np.copy(rgiObAct))
                for ob1 in rgiObAct:
                    tt1_obj = list(np.unique(rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt-1,:] == ob1])[1:])
                    if len(tt1_obj) == 0:
                        # this object ends here
                        rgiObActCP.remove(ob1)
                        continue
                    elif len(tt1_obj) == 1:
                        rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt,:] == tt1_obj[0]] = ob1
                    else:
                        VOL = [np.sum(rgiObjects2D_ACT[tt,:] == tt1_obj[jj]) for jj in range(len(tt1_obj))]
                        rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt,:] == tt1_obj[np.argmax(VOL)]] = ob1
                        tt1_obj.remove(tt1_obj[np.argmax(VOL)])
                        rgiObActCP = rgiObActCP + list(tt1_obj)

                # make sure that mergers are assigned the largest object
                for ob2 in rgiObActCP:
                    ttm1_obj = list(np.unique(rgiObjects2D_ACT[tt-1,rgiObjects2D_ACT[tt,:] == ob2])[1:])
                    if len(ttm1_obj) > 1:
                        VOL = [np.sum(rgiObjects2D_ACT[tt-1,:] == ttm1_obj[jj]) for jj in range(len(ttm1_obj))]
                        rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt,:] == ob2] = ttm1_obj[np.argmax(VOL)]


                # are there new object?
                NewObj = np.unique(rgiObjects2D_ACT[tt,:,:])[1:]
                NewObj = list(np.setdiff1d(NewObj,rgiObAct))
                if len(NewObj) != 0:
                    rgiObActCP = rgiObActCP + NewObj
                rgiObActCP = np.unique(rgiObActCP)
                rgiObAct = np.copy(rgiObActCP)

            rgiObjects2D_ACT[rgiObjects2D_ACT !=0] = np.copy(rgiObjects2D_ACT[rgiObjects2D_ACT !=0]+MaxOb)
            MaxOb = np.max(DATA)

            # save the new objects to the original object array
            TMP = np.copy(DATA[Objects[SelOb]])
            TMP[rgiObjects2D_ACT != 0] = rgiObjects2D_ACT[rgiObjects2D_ACT != 0]
            DATA[Objects[SelOb]] = np.copy(TMP)

    # clean up object matrix
    Unique = np.unique(DATA)[1:]
    Objects=ndimage.find_objects(DATA)
    rgiVolObj=np.array([np.sum(DATA[Objects[Unique[ob]-1]] == Unique[ob]) for ob in range(len(Unique))])
    TT = np.array([Objects[Unique[ob]-1][0].stop - Objects[Unique[ob]-1][0].start for ob in range(len(Unique))])

    # create final object array
    CY_objectsTMP=np.copy(DATA); CY_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiVolObj)):
        if TT[ob] >= MinTime/dT:
            CY_objectsTMP[DATA == Unique[ob]] = ii
            ii = ii + 1

    # lable the objects from 1 to N
    DATA_fin=np.copy(CY_objectsTMP); DATA_fin[:]=0
    Unique = np.unique(CY_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        DATA_fin[CY_objectsTMP == Unique[ob]] = ii
        ii = ii + 1

    return DATA_fin


# from https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("        "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# from - https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def DistanceCoord(Lo1,La1,Lo2,La2):

    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(La1)
    lon1 = radians(Lo1)
    lat2 = radians(La2)
    lon2 = radians(Lo2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))


# https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
import numpy as np
from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval



# ================================================================================================
# ================================================================================================

def MultiObjectIdentification(
    DATA_all,                      # matrix with data on common grid in the format [time,lat,lon,variable]
                                   # the variables are 'V850' [m/s], 'U850' [m/s], 'T850' K, 'Q850' g/kg,
                                   # 'SLP' [Pa], 'IVTE' [kg m-1 s-1], 'IVTN' [kg m-1 s-1], 'PR' [mm/time], 'BT' [K]]
                                   # this order must be followed
    Lon,                           # 2D longitude grid centers
    Lat,                           # 2D latitude grid spacing
    Time,                          # datetime vector of data
    dT,                            # integer - temporal frequency of data [hour]
    Mask,                          # mask with dimensions [lat,lon] defining analysis region
    DataName = '',                 # name of the common grid
    OutputFolder='',               # string containing the output directory path. Default is local directory
    SmoothSigmaP = 0,              # Gaussion std for precipitation smoothing
    Pthreshold = 2,                # precipitation threshold [mm/h]
    MinTimePR = 6,                 # minimum lifetime of precip. features in hours
    MinAreaPR = 5000,              # minimum area of precipitation features [km2]
    # minimum Moisture Stream
    MinTimeMS = 9,                 # minimum lifetime for moisture stream [hours]
    MinAreaMS = 100000,            # mimimum area of moisture stream [km2]
    MinMSthreshold = 0.13,         # treshold for moisture stream [g*m/g*s]
    # cyclone tracking
    MinTimeCY = 12,                # minimum livetime of cyclones [hours]
    MaxPresAnCY = -8,              # preshure thershold for cyclone anomaly [hPa]
    # anty cyclone tracking
    MinTimeACY = 12,               # minimum livetime of anticyclone [hours]
    MinPresAnACY = 6,              # preshure thershold for anti cyclone anomaly [hPa]
    # Frontal zones
    MinAreaFR = 50000,             # mimimum size of frontal zones [km2]
    # Cloud tracking setup
    SmoothSigmaC = 0,              # standard deviation of Gaussian filter for cloud tracking
    Cthreshold = 241,              # brightness temperature threshold for cloud tracking [K]
    MinTimeC = 9,                  # mimimum livetime of ice cloud shields [hours]
    MinAreaC = 40000,              # mimimum area of ice cloud shields [km2]
    # AR tracking
    IVTtrheshold = 500,            # Integrated water vapor transport threshold for AR detection [kg m-1 s-1]
    MinTimeIVT = 9,                # minimum livetime of ARs [hours]
    AR_MinLen = 2000,              # mimimum length of an AR [km]
    AR_Lat = 20,                   # AR centroids have to be poeward of this latitude
    AR_width_lenght_ratio = 2,     # mimimum length to width ratio of AR
    # TC detection
    TC_Pmin = 995,                 # mimimum pressure for TC detection [hPa]
    TC_lat_genesis = 35,           # maximum latitude for TC genesis [absolute degree latitude]
    TC_lat_max = 60,               # maximum latitude for TC existance [absolute degree latitude]
    TC_deltaT_core = 0,            # minimum degrees difference between TC core and surrounding [K]
    TC_T850min = 285,              # minimum temperature of TC core at 850hPa [K]
    TC_minBT = 241,                # minimum average cloud top brightness temperature [K]
    # MCs detection
    MCS_Minsize = 2500,            # minimum size of precipitation area [km2]
    MCS_minPR = 10,                 # minimum precipitation threshold [mm/h]
    CL_MaxT = 225,                 # minimum brightness temperature in ice shield [K]
    CL_Area = 40000,               # minimum cloud area size [km2]
    MCS_minTime = 4                # minimum lifetime of MCS [hours]
    ):

    Variables = ['V', 'U', 'T', 'Q', 'SLP', 'IVTE', 'IVTN', 'PR', 'BT']
    # calculate grid spacing assuming regular lat/lon grid
    EarthCircum = 40075000 #[m]
    dLat = np.copy(Lon); dLat[:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))
    dLon = np.copy(Lon)
    for la in range(Lat.shape[0]):
        dLon[la,:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))*np.cos(np.deg2rad(Lat[la,0]))
    Gridspacing = np.mean(np.append(dLat[:,:,None],dLon[:,:,None], axis=2))
    Area = dLat*dLon
    Area[Area < 0] = 0

    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    StartDay = Time[0]
    SetupString = 'dt-'+str(dT)+'h_PRTr-'+str(Pthreshold)+'_PRS-'+str(SmoothSigmaP)+'_ARt-'+str(MinTimeIVT)+'_ARL-'+str(AR_MinLen)+'_CYt-'+str(MinTimeCY)+'_FRA-'+str(MinAreaFR)+'_CLS-'+str(SmoothSigmaC)+'_CLT-'+str(Cthreshold)+'_CLt-'+str(MinTimeC)+'_CLA-'+str(MinAreaC)+'_IVTTr-'+str(IVTtrheshold)+'_IVTt-'+str(MinTimeIVT)
    NCfile = OutputFolder + str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+'_ObjectMasks_'+SetupString+'.nc'
    FrontMask = np.copy(Mask)
    FrontMask[np.abs(Lat) < 10] = 0

    # connect over date line?
    if (Lon[0,0] < 176) & (Lon[0,-1] > 176):
        connectLon= 1
    else:
        connectLon= 0


    print('    Derive nescessary varialbes for feature indentification')
    import time
    start = time.perf_counter()
    # 11111111111111111111111111111111111111111111111111
    # calculate vapor transport on pressure level
    VapTrans = ((DATA_all[:,:,:,Variables.index('U')]*DATA_all[:,:,:,Variables.index('Q')])**2 + (DATA_all[:,:,:,Variables.index('V')]*DATA_all[:,:,:,Variables.index('Q')])**2)**(1/2)

    # 22222222222222222222222222222222222222222222222222
    # Frontal Detection according to https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073662
    UU = DATA_all[:,:,:,Variables.index('U')]
    VV = DATA_all[:,:,:,Variables.index('V')]
    dx = dLon
    dy = dLat
    du = np.gradient( UU )
    dv = np.gradient( VV )
    PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
    TK = DATA_all[:,:,:,Variables.index('T')]
    vgrad = np.gradient(TK, axis=(1,2))
    Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

    Fstar = PV * Tgrad

    Tgrad_zero = 0.45 #*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
    import metpy.calc as calc
    from metpy.units import units
    CoriolisPar = calc.coriolis_parameter(np.deg2rad(Lat))
    Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))

    # # 3333333333333333333333333333333333333333333333333333
    # # Cyclone identification based on pressure annomaly threshold

    SLP = DATA_all[:,:,:,Variables.index('SLP')]/100.
    if np.sum(np.isnan(SLP)) == 0:
        # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
        SLP_smooth = ndimage.uniform_filter(SLP, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        # smoothign over 3000 x 3000 km and 78 hours
        SLPsmoothAn = ndimage.uniform_filter(SLP, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    else:
        # this code takes care of the smoothing of fields that contain NaN values
        # from - https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        U=SLP.copy()               # random array...
        V=SLP.copy()
        V[np.isnan(U)]=0
        VV = ndimage.uniform_filter(V, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
        W=0*U.copy()+1
        W[np.isnan(U)]=0
        WW=ndimage.uniform_filter(W, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
        SLPsmoothAn=VV/WW

        VV = ndimage.uniform_filter(V, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        WW=ndimage.uniform_filter(W, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        SLP_smooth = VV/WW

    SLP_Anomaly = SLP_smooth-SLPsmoothAn
    SLP_Anomaly[:,Mask == 0] = np.nan
    # plt.contour(SLP_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = SLP_Anomaly < MaxPresAnCY # 10 hPa depression | original setting was 12
    HighPressure_annomaly = SLP_Anomaly > MinPresAnACY #12

    # from - https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    sigma=10.0                  # standard deviation for Gaussian kernel
    truncate=10.                # truncate filter at this many sigmas

    U=SLP.copy()               # random array...

    V=SLP.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=[sigma,sigma,sigma],truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=[sigma,sigma,sigma],truncate=truncate)

    Z=VV/WW


    # 4444444444444444444444444444444444444444444444444444444
    # calculate IVT
    IVT = ((DATA_all[:,:,:,Variables.index('IVTE')])**2+np.abs(DATA_all[:,:,:,Variables.index('IVTN')])**2)**0.5

    # Mask data outside of Focus domain
    DATA_all[:,Mask == 0,:] = np.nan
    Pressure_anomaly[:,Mask == 0] = np.nan
    HighPressure_annomaly[:,Mask == 0] = np.nan
    Frontal_Diagnostic[:,Mask == 0] = np.nan
    VapTrans[:,Mask == 0] = np.nan
    SLP[:,Mask == 0] = np.nan

    end = time.perf_counter()
    timer(start, end)


    ### Perform Feature Tracking
    #### ------------------------
    print('        track  moisture streams in extratropics')
    potARs = (VapTrans > MinMSthreshold)
    rgiObjectsAR, nr_objectsUD = ndimage.label(potARs, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsAR)

    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1:]][rgiObjectsAR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsAR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # create final object array
    MS_objectsTMP=np.copy(rgiObjectsAR); MS_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaMS*1000**2, np.ones(int(MinTimeMS/dT)), mode='valid'))
        if (AreaTest == int(MinTimeMS/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimeMS/dT)):
            MS_objectsTMP[rgiObjectsAR == (ob+1)] = ii
            ii = ii + 1
    # lable the objects from 1 to N
    MS_objects=np.copy(rgiObjectsAR); MS_objects[:]=0
    Unique = np.unique(MS_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        MS_objects[MS_objectsTMP == Unique[ob]] = ii
        ii = ii + 1

    print('        break up long living MS objects that have many elements')
    MS_objects = BreakupObjects(MS_objects,
                                int(MinTimeMS/dT),
                                dT)

    if connectLon == 1:
        print('        connect MS objects over date line')
        MS_objects = ConnectLon(MS_objects)


    grMSs = ObjectCharacteristics(MS_objects, # feature object file
                                 VapTrans,         # original file used for feature detection
                                 OutputFolder+'MS850_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 MinTime=int(MinTimeMS/dT))      # minimum livetime in hours


    #### ------------------------
    print('    track  IVT')
    start = time.perf_counter()

    potIVTs = (IVT > IVTtrheshold)
    rgiObjectsIVT, nr_objectsUD = ndimage.label(potIVTs, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsIVT)
    # rgiVolObj=np.array([np.sum(rgiObjectsIVT[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    TT_CY = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)])

    # create final object array
    IVT_objectsTMP=np.copy(rgiObjectsIVT); IVT_objectsTMP[:]=0
    ii = 1
    for ob in range(len(TT_CY)):
        if TT_CY[ob] >= int(MinTimeIVT/dT):
            IVT_objectsTMP[rgiObjectsIVT == (ob+1)] = ii
            ii = ii + 1
    # lable the objects from 1 to N
    IVT_objects=np.copy(rgiObjectsIVT); IVT_objects[:]=0
    Unique = np.unique(IVT_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        IVT_objects[IVT_objectsTMP == Unique[ob]] = ii
        ii = ii + 1

    print('        break up long living IVT objects that have many elements')
    IVT_objects = BreakupObjects(IVT_objects,
                                 int(MinTimeIVT/dT),
                                dT)

    if connectLon == 1:
        print('        connect IVT objects over date line')
        IVT_objects = ConnectLon(IVT_objects)

    grIVTs = ObjectCharacteristics(IVT_objects, # feature object file
                                 VapTrans,         # original file used for feature detection
                                 OutputFolder+'IVT_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 MinTime=int(MinTimeIVT/dT))      # minimum livetime in hours
    end = time.perf_counter()
    timer(start, end)

    print('    check if MSs quallify as ARs')
    start = time.perf_counter()
    if IVT_objects.max() != 0:
        AR_obj = np.copy(IVT_objects); AR_obj[:] = 0.
        Objects=ndimage.find_objects(IVT_objects.astype(int))
    else:
        AR_obj = np.copy(MS_objects); AR_obj[:] = 0.
        Objects=ndimage.find_objects(MS_objects.astype(int))
        IVT_objects = MS_objects
    aa=1
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = IVT_objects[Objects[ii]] == ii+1
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)

        OBJ_max_len = np.zeros((ObjACT.shape[0]))
        for tt in range(ObjACT.shape[0]):
            PointsObj = np.append(LonObj[ObjACT[tt,:,:]==1][:,None], LatObj[ObjACT[tt,:,:]==1][:,None], axis=1)
            try:
                Hull = scipy.spatial.ConvexHull(np.array(PointsObj))
            except:
                continue
            XX = []; YY=[]
            for simplex in Hull.simplices:
    #                 plt.plot(PointsObj[simplex, 0], PointsObj[simplex, 1], 'k-')
                XX = XX + [PointsObj[simplex, 0][0]]
                YY = YY + [PointsObj[simplex, 1][0]]

            points = [[XX[ii],YY[ii]] for ii in range(len(YY))]
            BOX = minimum_bounding_rectangle(np.array(PointsObj))

            DIST = np.zeros((3))
            for rr in range(3):
                DIST[rr] = DistanceCoord(BOX[rr][0],BOX[rr][1],BOX[rr+1][0],BOX[rr+1][1])
            OBJ_max_len[tt] = np.max(DIST)
            if OBJ_max_len[tt] <= AR_MinLen:
                ObjACT[tt,:,:] = 0
            else:
                rgiCenter = np.round(ndimage.measurements.center_of_mass(ObjACT[tt,:,:])).astype(int)
                LatCent = LatObj[rgiCenter[0],rgiCenter[1]]
                if np.abs(LatCent) < AR_Lat:
                    ObjACT[tt,:,:] = 0
            # check width to lenght ratio
            if DIST.max()/DIST.min() < AR_width_lenght_ratio:
                ObjACT[tt,:,:] = 0
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = aa
        ObjACT = ObjACT + AR_obj[Objects[ii]]
        AR_obj[Objects[ii]] = ObjACT
        aa=aa+1

    end = time.perf_counter()
    timer(start, end)


    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track cyclones')
    start = time.perf_counter()
    # Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    Pressure_anomaly[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(Pressure_anomaly,structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsUD)
    rgiVolObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    TT_CY = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)])

    # create final object array
    CY_objectsTMP=np.copy(rgiObjectsUD); CY_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiVolObj)):
        if TT_CY[ob] >= int(MinTimeCY/dT):
            CY_objectsTMP[rgiObjectsUD == (ob+1)] = ii
            ii = ii + 1

    # lable the objects from 1 to N
    CY_objects=np.copy(rgiObjectsUD); CY_objects[:]=0
    Unique = np.unique(CY_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        CY_objects[CY_objectsTMP == Unique[ob]] = ii
        ii = ii + 1

    print('        break up long living CY objects that have many elements')
    CY_objects = BreakupObjects(CY_objects,
                               int(MinTimeCY/dT),
                               dT)

    if connectLon == 1:
        print('        connect cyclones objects over date line')
        CY_objects = ConnectLon(CY_objects)

    grCyclonesPT = ObjectCharacteristics(CY_objects, # feature object file
                                     SLP,         # original file used for feature detection
                                     OutputFolder+'CY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     MinTime=int(MinTimeCY/dT))
    end = time.perf_counter()
    timer(start, end)



    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track anti-cyclones')
    start = time.perf_counter()
    HighPressure_annomaly[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(HighPressure_annomaly,structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsUD)
    rgiVolObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    TT_ACY = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)])

    # create final object array
    ACY_objectsTMP=np.copy(rgiObjectsUD); ACY_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiVolObj)):
        if TT_ACY[ob] >= MinTimeACY:
    #     if rgiVolObj[ob] >= MinVol:
            ACY_objectsTMP[rgiObjectsUD == (ob+1)] = ii
            ii = ii + 1

    # lable the objects from 1 to N
    ACY_objects=np.copy(rgiObjectsUD); ACY_objects[:]=0
    Unique = np.unique(ACY_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        ACY_objects[ACY_objectsTMP == Unique[ob]] = ii
        ii = ii + 1

    print('        break up long living ACY objects that have many elements')
    ACY_objects = BreakupObjects(ACY_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        # connect objects over date line
        ACY_objects = ConnectLon(ACY_objects)

    grACyclonesPT = ObjectCharacteristics(ACY_objects, # feature object file
                                     SLP,         # original file used for feature detection
                                     OutputFolder+'ACY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     MinTime=int(MinTimeCY/dT))
    end = time.perf_counter()
    timer(start, end)


    ### Identify Frontal regions
    # ------------------------
    print('    identify frontal zones')
    start = time.perf_counter()
    rgiObj_Struct_Fronts=np.zeros((3,3,3)); rgiObj_Struct_Fronts[1,:,:]=1

    Frontal_Diagnostic = np.abs(Frontal_Diagnostic)
    Frontal_Diagnostic[:,FrontMask == 0] = 0
    Fmask = (Frontal_Diagnostic > 1)

    rgiObjectsUD, nr_objectsUD = ndimage.label(Fmask,structure=rgiObj_Struct_Fronts)
    print('        '+str(nr_objectsUD)+' object found')

    # # calculate object size
    Objects=ndimage.find_objects(rgiObjectsUD)
    rgiAreaObj = np.array([np.sum(Area[Objects[ob][1:]][rgiObjectsUD[Objects[ob]][0,:,:] == ob+1]) for ob in range(nr_objectsUD)])

    # rgiAreaObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    # create final object array
    FR_objects=np.copy(rgiObjectsUD)
    TooSmall = np.where(rgiAreaObj < MinAreaFR*1000**2)
    FR_objects[np.isin(FR_objects, TooSmall[0]+1)] = 0

#     FR_objects=np.copy(rgiObjectsUD); FR_objects[:]=0
#     ii = 1
#     for ob in range(len(rgiAreaObj)):
#         if rgiAreaObj[ob] >= MinAreaFR*1000**2:
#             FR_objects[rgiObjectsUD == (ob+1)] = ii
#             ii = ii + 1
    end = time.perf_counter()
    timer(start, end)


    # ------------------------
    print('    track  precipitation')
    start = time.perf_counter()
    PRsmooth=gaussian_filter(DATA_all[:,:,:,Variables.index('PR')], sigma=(0,SmoothSigmaP,SmoothSigmaP))
    PRmask = (PRsmooth >= Pthreshold*dT)
    rgiObjectsPR, nr_objectsUD = ndimage.label(PRmask, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' precipitation object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsPR = ConnectLon(rgiObjectsPR)

    # remove None objects
    Objects=ndimage.find_objects(rgiObjectsPR)
    rgiVolObj=np.array([np.sum(rgiObjectsPR[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    ZERO_V =  np.where(rgiVolObj == 0)
    if len(ZERO_V[0]) > 0:
        Dummy = [slice(0, 1, None), slice(0, 1, None), slice(0, 1, None)]
        Objects = np.array(Objects)
        for jj in ZERO_V[0]:
            Objects[jj] = Dummy

    # Remove objects that are too small or short lived
    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsPR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsPR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])
    # create final object array
    PR_objects=np.copy(rgiObjectsPR); PR_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaPR*1000**2, np.ones(int(MinTimePR/dT)), mode='valid'))
        if (AreaTest == int(MinTimePR/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimePR/dT)):
            PR_objects[rgiObjectsPR == (ob+1)] = ii
            ii = ii + 1

    if connectLon == 1:
        print('        connect precipitation objects over date line')
        PR_objects = ConnectLon(PR_objects)

    grPRs = ObjectCharacteristics(PR_objects, # feature object file
                                 DATA_all[:,:,:,Variables.index('PR')],         # original file used for feature detection
                                 OutputFolder+'PR_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 MinTime=int(MinTimePR/dT))      # minimum livetime in hours
    end = time.perf_counter()
    timer(start, end)


    # ------------------------
    print('    track  clouds')
    start = time.perf_counter()
    Csmooth=gaussian_filter(DATA_all[:,:,:,Variables.index('BT')], sigma=(0,SmoothSigmaC,SmoothSigmaC))
    Cmask = (Csmooth <= Cthreshold)
    rgiObjectsC, nr_objectsUD = ndimage.label(Cmask, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' cloud object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsC = ConnectLon(rgiObjectsC)

    # minimum cloud volume
    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsC)

    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsC[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsC[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # rgiVolObjC=np.array([np.sum(rgiObjectsC[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])

    # create final object array
    C_objects=np.copy(rgiObjectsC); C_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaC*1000**2, np.ones(int(MinTimeC/dT)), mode='valid'))
        if (AreaTest == int(MinTimeC/dT)) & (len(rgiAreaObj[ob]) >=int(MinTimeC/dT)):
        # if rgiVolObjC[ob] >= MinAreaC:
            C_objects[rgiObjectsC == (ob+1)] = ii
            ii = ii + 1

    print('        break up long living cloud shield objects that have many elements')
    C_objects = BreakupObjects(C_objects,
                                int(MinTimeC/dT),
                                dT)

    if connectLon == 1:
        print('        connect cloud objects over date line')
        C_objects = ConnectLon(C_objects)

    grCs = ObjectCharacteristics(C_objects, # feature object file
                                 DATA_all[:,:,:,Variables.index('BT')],         # original file used for feature detection
                                 OutputFolder+'Clouds_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 MinTime=int(MinTimeC/dT))      # minimum livetime in hours
    end = time.perf_counter()
    timer(start, end)


    # ------------------------
    print('    check if pr objects quallify as MCS')
    start = time.perf_counter()
    # check if precipitation object is from an MCS
    Objects=ndimage.find_objects(PR_objects.astype(int))
    MCS_obj = np.copy(PR_objects); MCS_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = PR_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2:
            continue
        Cloud_ACT = np.copy(C_objects[Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        Area_ACT = Area[Objects[ii][1],Objects[ii][2]]
        PR_ACT = DATA_all[:,:,:,Variables.index('PR')][Objects[ii]]

        PR_Size = np.array([np.sum(Area_ACT[ObjACT[tt,:,:] >0]) for tt in range(ObjACT.shape[0])])
        PR_MAX = np.array([np.max(PR_ACT[tt,ObjACT[tt,:,:] >0]) if len(PR_ACT[tt,ObjACT[tt,:,:]>0]) > 0 else 0 for tt in range(ObjACT.shape[0])])
        # Get cloud shield
        rgiCL_obj = np.delete(np.unique(Cloud_ACT[ObjACT > 0]),0)
        if len(rgiCL_obj) == 0:
            # no deep cloud shield is over the precipitation
            continue
        CL_OB_TMP = C_objects[Objects[ii][0]]
        CL_TMP = DATA_all[:,:,:,Variables.index('BT')][Objects[ii][0]]
        CLOUD_obj_act = np.in1d(CL_OB_TMP.flatten(), rgiCL_obj).reshape(CL_OB_TMP.shape)
        Cloud_Size = np.array([np.sum(Area[CLOUD_obj_act[tt,:,:] >0]) for tt in range(CLOUD_obj_act.shape[0])])
        # min temperatur must be taken over precip area
        CL_ob_pr = C_objects[Objects[ii]]
        CL_BT_pr = DATA_all[:,:,:,Variables.index('BT')][Objects[ii]]
        Cloud_MinT = np.array([np.min(CL_BT_pr[tt,CL_ob_pr[tt,:,:] >0]) if len(CL_ob_pr[tt,CL_ob_pr[tt,:,:] >0]) > 0 else 0 for tt in range(CL_ob_pr.shape[0])])
        # is precipitation associated with AR?
        AR_ob = np.copy(AR_obj[Objects[ii]])
        AR_ob[:,LatObj < 25] = 0 # only consider ARs in mid- and hight latitudes
        AR_test = np.sum(AR_ob > 0, axis=(1,2))

        # Test if object is an MCS
        MCS_TEST = (Cloud_Size >= CL_Area) & (Cloud_MinT <= CL_MaxT) & (PR_Size >= MCS_Minsize) & (PR_MAX >= MCS_minPR*dT) & (AR_test == 0)

        # assign unique object numbers
        ObjACT = np.array(ObjACT).astype(int)
        ObjACT[ObjACT == 1] = ii+1

        # remove all precip that is associated with ARs
        ObjACT[AR_test > 0] = 0

        # PR area defines MCS area and precipitation
        window_length = int(MCS_minTime/dT)
        cumulative_sum = np.cumsum(np.insert(MCS_TEST, 0, 0))
        moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
        if np.max(moving_averages) == 1:
            TMP = np.copy(MCS_obj[Objects[ii]])
            TMP = TMP + ObjACT
            MCS_obj[Objects[ii]] = TMP
        else:
            continue
    end = time.perf_counter()
    timer(start, end)


    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    print('    Check if cyclones qualify as TCs')
    start = time.perf_counter()
    TC_Tracks = {}
    TC_Time = {}
    aa=1
    # check if cyclone is tropical
    Objects=ndimage.find_objects(CY_objects.astype(int))
    TC_obj = np.copy(CY_objects); TC_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = CY_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2*8:
            continue
        T_ACT = np.copy(TK[Objects[ii]])
        SLP_ACT = np.copy(SLP[Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)
            SLP_ACT = np.roll(SLP_ACT, int(ObjACT.shape[2]/2), axis=2)
        # Calculate low pressure center track
        SLP_ACT[ObjACT == 0] = 999999999.
        Track_ACT = np.array([np.argwhere(SLP_ACT[tt,:,:] == np.nanmin(SLP_ACT[tt,:,:]))[0] for tt in range(ObjACT.shape[0])])
        LatLonTrackAct = np.array([(LatObj[Track_ACT[tt][0],Track_ACT[tt][1]],LonObj[Track_ACT[tt][0],Track_ACT[tt][1]]) for tt in range(ObjACT.shape[0])])
        if np.min(np.abs(LatLonTrackAct[:,0])) > TC_lat_genesis:
            ObjACT[:] = 0
            continue
        else:

            # has the cyclone a warm core?
            DeltaTCore = np.zeros((ObjACT.shape[0])); DeltaTCore[:] = np.nan
            T850_core = np.copy(DeltaTCore)
            for tt in range(ObjACT.shape[0]):
                T_cent = np.mean(T_ACT[tt,Track_ACT[tt,0]-1:Track_ACT[tt,0]+2,Track_ACT[tt,1]-1:Track_ACT[tt,1]+2])
                T850_core[tt] = T_cent
                T_Cyclone = np.mean(T_ACT[tt,ObjACT[tt,:,:] != 0])
    #                     T_Cyclone = np.mean(T_ACT[tt,MassC[0]-5:MassC[0]+6,MassC[1]-5:MassC[1]+6])
                DeltaTCore[tt] = T_cent-T_Cyclone
            # smooth the data
            DeltaTCore = gaussian_filter(DeltaTCore,1)
            WarmCore = DeltaTCore > TC_deltaT_core

            if np.sum(WarmCore) < 8:
                continue
            ObjACT[WarmCore == 0,:,:] = 0
            # is the core temperature warm enough
            ObjACT[T850_core < TC_T850min,:,:] = 0


            # TC must have pressure of less 980 hPa
            MinPress = np.min(SLP_ACT, axis=(1,2))
            if np.sum(MinPress < TC_Pmin) < 8:
                continue

            # is the cloud shield cold enough?
            PR_objACT = np.copy(PR_objects[Objects[ii]])
            BT_act = np.copy(DATA_all[:,:,:,Variables.index('BT')][Objects[ii]])
            BT_objMean = np.zeros((BT_act.shape[0])); BT_objMean[:] = np.nan
            for tt in range(len(BT_objMean)):
                try:
                    BT_objMean[tt] = np.nanmean(BT_act[tt,PR_objACT[tt,:,:] != 0])
                except:
                    continue

        # remove pieces of the track that are not TCs
        TCcheck = (T850_core > TC_T850min) & (WarmCore == 1) & (MinPress < TC_Pmin) #(BT_objMean < TC_minBT)
        LatLonTrackAct[TCcheck == False,:] = np.nan

        Max_LAT = (np.abs(LatLonTrackAct[:,0]) >  TC_lat_max)
        LatLonTrackAct[Max_LAT,:] = np.nan

        if np.sum(~np.isnan(LatLonTrackAct[:,0])) == 0:
            continue

        # check if cyclone genesis is over water; each re-emergence of TC is new genesis
        resultLAT = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,0], np.isnan) if not k]
        resultLON = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,1], np.isnan) if not k]
        LS_genesis = np.zeros((len(resultLAT))); LS_genesis[:] = np.nan
        for jj in range(len(resultLAT)):
            LS_genesis[jj] = is_land(resultLON[jj][0],resultLAT[jj][0])
        if np.max(LS_genesis) == 1:
            for jj in range(len(LS_genesis)):
                if LS_genesis[jj] == 1:
                    SetNAN = np.isin(LatLonTrackAct[:,0],resultLAT[jj])
                    LatLonTrackAct[SetNAN,:] = np.nan

        # make sure that only TC time slizes are considered
        ObjACT[np.isnan(LatLonTrackAct[:,0]),:,:] = 0

        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = aa

        ObjACT = ObjACT + TC_obj[Objects[ii]]
        TC_obj[Objects[ii]] = ObjACT
        TC_Tracks[str(aa)] = LatLonTrackAct
        aa=aa+1

    end = time.perf_counter()
    timer(start, end)

    print(' ')
    print('Save the object masks into a joint netCDF')
    start = time.perf_counter()
    # ============================
    # Write NetCDF
    iTime = np.array((Time - Time[0]).total_seconds()).astype('int')

    dataset = Dataset(NCfile,'w',format='NETCDF4_CLASSIC')
    yc = dataset.createDimension('yc', Lat.shape[0])
    xc = dataset.createDimension('xc', Lat.shape[1])
    time = dataset.createDimension('time', None)

    times = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('yc','xc',))
    lon = dataset.createVariable('lon', np.float32, ('yc','xc',))
    PR_real = dataset.createVariable('PR', np.float32,('time','yc','xc'))
    PR_obj = dataset.createVariable('PR_Objects', np.float32,('time','yc','xc'))
    MCSs = dataset.createVariable('MCS_Objects', np.float32,('time','yc','xc'))
    Cloud_real = dataset.createVariable('BT', np.float32,('time','yc','xc'))
    Cloud_obj = dataset.createVariable('BT_Objects', np.float32,('time','yc','xc'))
    FR_real = dataset.createVariable('FR', np.float32,('time','yc','xc'))
    FR_obj = dataset.createVariable('FR_Objects', np.float32,('time','yc','xc'))
    CY_real = dataset.createVariable('CY', np.float32,('time','yc','xc'))
    CY_obj = dataset.createVariable('CY_Objects', np.float32,('time','yc','xc'))
    TCs = dataset.createVariable('TC_Objects', np.float32,('time','yc','xc'))
    ACY_obj = dataset.createVariable('ACY_Objects', np.float32,('time','yc','xc'))
    MS_real = dataset.createVariable('MS', np.float32,('time','yc','xc'))
    MS_obj = dataset.createVariable('MS_Objects', np.float32,('time','yc','xc'))
    IVT_real = dataset.createVariable('IVT', np.float32,('time','yc','xc'))
    IVT_obj = dataset.createVariable('IVT_Objects', np.float32,('time','yc','xc'))
    ARs = dataset.createVariable('AR_Objects', np.float32,('time','yc','xc'))
    SLP_real = dataset.createVariable('SLP', np.float32,('time','yc','xc'))
    T_real = dataset.createVariable('T850', np.float32,('time','yc','xc'))

    times.calendar = "standard"
    times.units = "seconds since "+str(Time[0].year)+"-"+str(Time[0].month).zfill(2)+"-"+str(Time[0].day).zfill(2)+" "+str(Time[0].hour).zfill(2)+":"+str(Time[0].minute).zfill(2)+":00"
    times.standard_name = "time"
    times.long_name = "time"

    lat.long_name = "latitude" ;
    lat.units = "degrees_north" ;
    lat.standard_name = "latitude" ;

    lon.long_name = "longitude" ;
    lon.units = "degrees_east" ;
    lon.standard_name = "longitude" ;

    PR_real.coordinates = "lon lat"
    PR_obj.coordinates = "lon lat"
    MCSs.coordinates = "lon lat"
    FR_real.coordinates = "lon lat"
    FR_obj.coordinates = "lon lat"
    CY_real.coordinates = "lon lat"
    CY_obj.coordinates = "lon lat"
    ACY_obj.coordinates = "lon lat"
    SLP_real.coordinates = "lon lat"
    T_real.coordinates = "lon lat"
    Cloud_real.coordinates = "lon lat"
    Cloud_obj.coordinates = "lon lat"
    MS_real.coordinates = "lon lat"
    MS_obj.coordinates = "lon lat"
    IVT_real.coordinates = "lon lat"
    IVT_obj.coordinates = "lon lat"
    ARs.coordinates = "lon lat"
    TCs.coordinates = "lon lat"

    lat[:] = Lat
    lon[:] = Lon
    PR_real[:] = DATA_all[:,:,:,Variables.index('PR')]
    PR_obj[:] = PR_objects
    MCSs[:] = MCS_obj
    FR_real[:] = Frontal_Diagnostic
    FR_obj[:] = FR_objects
    CY_real[:] = SLP_Anomaly
    CY_obj[:] = CY_objects
    TCs[:] = TC_obj
    ACY_obj[:] = ACY_objects
    SLP_real[:] = SLP
    T_real[:] = TK
    MS_real[:] = VapTrans
#     MS_obj[:] = MS_objects
    IVT_real[:] = IVT
    IVT_obj[:] = IVT_objects
    ARs[:] = AR_obj
    Cloud_real[:] = DATA_all[:,:,:,Variables.index('BT')]
    Cloud_obj[:] = C_objects
    times[:] = iTime

    dataset.close()
    print('Saved: '+NCfile)
    import time
    end = time.perf_counter()
    timer(start, end)

    ### SAVE THE TC TRACKS TO PICKL FILE
    # ============================
    a_file = open(OutputFolder+str(Time[0].year)+str(Time[0].month).zfill(2)+'_TCs_tracks.pkl', "wb")
    pickle.dump(TC_Tracks, a_file)
    a_file.close()
    # CYCLONES[YYYY+MM] = TC_Tracks


####====================================
# function to read imerg data
def readIMERG(TimeHH,
            Lon,
            Lat,
            iNorth,
            iEast,
            iSouth,
            iWest,
            dT):
    PR_DATA = np.zeros((len(TimeHH),Lon.shape[0],Lon.shape[1])); PR_DATA[:]=np.nan

    for tt in tqdm(range(len(TimeHH))):
    #     print('    read '+str(TimeHH[tt]))
        PR_IM = np.zeros((2,Lon.shape[0],Lon.shape[1]))
        for hh in range(2):
            if hh == 0:
                MINact = '00'
            else:
                MINact = '30'
            DATESTAMP = str(TimeHH[tt].year)+str(TimeHH[tt].month).zfill(2)+str(TimeHH[tt].day).zfill(2)
            TIMESTAMP = str(TimeHH[tt].hour).zfill(2)+MINact+'00'
            NUMBER1 = str(int(TIMESTAMP)+2959).zfill(6)
            FILEact = 'CONUS_3B-HHR.MS.MRG.3IMERG.'+DATESTAMP+'-S'+TIMESTAMP+'-E'+NUMBER1+'.*.V06B.nc' #'.nc3'
            try:
                FILEact = glob.glob(FILEact)[0]
            except:
                stop()
            # for instructions see - https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Read%20IMERG%20Data%20Using%20Python
            f = h5py.File(FILEact, 'r')
            PR_IM[hh,:,:] = f['Grid/precipitationCal'][0].T[iSouth:iNorth,iWest:iEast]
        PR_IM = np.mean(PR_IM, axis=0)
        PR_DATA[tt,:,:] = PR_IM

    # resampple data to common time period
    PR_DATA = np.sum(np.reshape(PR_DATA,(int(PR_DATA.shape[0]/dT),dT,PR_DATA.shape[1],PR_DATA.shape[2])), axis=1)

    return PR_DATA


####====================================
# function to read MERGIR data
def readMERGIR(TimeBT,
                Lon,
                Lat,
                dT,
                FocusRegion):
    # Read Brightness temperature from MERGIR
    # -----------
    ncfile = Dataset('CONUS_merg_2016070100_4km-pixel.nc4')
    LonG=np.squeeze(ncfile.variables["lon"])
    LatG=np.squeeze(ncfile.variables["lat"])
    ncfile.close()
    LonG,LatG = np.meshgrid(LonG,LatG)

    iNorth = np.argmin(np.abs(LatG[:,0] - FocusRegion[0]))
    iSouth = np.argmin(np.abs(LatG[:,0] - FocusRegion[2]))+1
    iEast = np.argmin(np.abs(LonG[0,:] - FocusRegion[1]))+1
    iWest = np.argmin(np.abs(LonG[0,:] - FocusRegion[3]))
    print(iNorth,iSouth,iWest,iEast)

    LonG = LonG[iSouth:iNorth,iWest:iEast]
    LatG = LatG[iSouth:iNorth,iWest:iEast]

    # Remap Gridsat to Target Grid
    points=np.array([LonG.flatten(), LatG.flatten()]).transpose()
    if os.path.isfile('Regrid_MERGIR_to_TarGrid.npz') == False:
        vtx, wts = interp_weights(points, np.append(Lon.flatten()[:,None], Lat.flatten()[:,None], axis=1))
        np.savez('Regrid_MERGIR_to_TarGrid.npz',vtx=vtx, wts=wts)
    DATA=np.load('Regrid_MERGIR_to_TarGrid.npz')
    vtxGS=DATA['vtx']
    wtsGS=DATA['wts']
    # Define region where MERGIR is valid
    rgrGridCells=[(Lon.ravel()[ii],Lat.ravel()[ii]) for ii in range(len(Lon.ravel()))]
    rgrSRactP=np.zeros((Lon.shape[0]*Lon.shape[1]))
    first = np.append(LonG[0,:][:,None],LatG[0,:][:,None], axis=1)
    second = np.append(LonG[:,0][:,None],LatG[:,0][:,None], axis=1)
    third = np.append(LonG[-1,:][:,None],LatG[-1,:][:,None], axis=1)
    fourth = np.append(LonG[:,-1][:,None],LatG[:,-1][:,None], axis=1)
    array_tuple = (first, second, third[::-1], fourth[::-1])
    ctr = np.vstack(array_tuple)
    rgrSRactP=np.zeros((Lon.shape[0]*Lon.shape[1]))
    grPRregion=mplPath.Path(ctr)
    TMP=np.array(grPRregion.contains_points(rgrGridCells))
    rgrSRactP[TMP == 1]=1
    rgrSRactG=np.reshape(rgrSRactP, (Lat.shape[0], Lat.shape[1]))

    CLOUD_DATA = np.zeros((len(TimeBT),Lon.shape[0],Lon.shape[1])); CLOUD_DATA[:]=np.nan
    for tt in tqdm(range(0,len(TimeBT))):
    #     print('    read '+str(TimeBT[tt]))
        TimeACT = TimeBT[tt]
        YYYY = str(TimeACT.year)
        MM = str(TimeACT.month).zfill(2)
        DD = str(TimeACT.day).zfill(2)
        HH = str(TimeACT.hour).zfill(2)

        # Read Gridsat
        FILE= 'CONUS_merg_'+YYYY+MM+DD+HH+'_4km-pixel.nc4'
        ncfile = Dataset(FILE)
        irwin_cdr=np.squeeze(ncfile.variables["Tb"][:,iSouth:iNorth,iWest:iEast])
        ncfile.close()
        irwin_cdr[irwin_cdr < 0] = np.nan
        irwin_cdr = np.nanmedian(irwin_cdr, axis=0)
    #     irwin_cdr[rgrSRactP == 0] = np.nan

        # GRIDSAT to ERA5
        valuesi=interpolate(irwin_cdr.flatten(), vtxGS, wtsGS)
        PRIM_on_GS=valuesi.reshape(Lat.shape[0],Lat.shape[1])
        PRIM_on_GS[PRIM_on_GS < 0]=np.nan
    #     PRIM_on_GS[rgrSRactPIM == 0] = np.nan
    #     PRIM_on_GS[rgrSRactP == 0] = np.nan
        CLOUD_DATA[tt,:,:] = PRIM_on_GS

    CLOUD_DATA[:,rgrSRactG == 0] =np.nan

    return CLOUD_DATA


#### ============================================================================================================
# function to perform MCS tracking
def MCStracking(DATA_all,
                Time,
                Lon,
                Lat,
                Variables,
                dT,                        # time step of data in hours
                SmoothSigmaP = 0,          # Gaussion std for precipitation smoothing
                Pthreshold = 2,            # precipitation threshold [mm/h]
                MinTimePR = 3,             # minum lifetime of PR feature in hours
                MinAreaPR = 5000,          # minimum area of precipitation feature in km2
                # Brightness temperature (Tb) tracking setup
                SmoothSigmaC = 0,          # Gaussion std for Tb smoothing
                Cthreshold = 241,          # minimum Tb of cloud shield
                MinTimeC = 9,              # minium lifetime of cloud shield in hours
                MinAreaC = 40000,          # minimum area of cloud shield in km2
                # MCs detection
                MCS_Minsize = 5000,        # km2
                MCS_minPR = 10,            # minimum max precipitation in mm/h
                MCS_MinPeakPR = 10,        # Minimum lifetime peak of MCS precipitation
                CL_MaxT = 225,             # minimum brightness temperature
                CL_Area = 40000,           # min cloud area size in km2
                MCS_minTime = 4,           # minimum time step
                NCfile = 'CONUS-MCS-tracking.nc'):

    # Calculate some parameters for the tracking
    EarthCircum = 40075000 #[m]
    dLat = np.copy(Lon); dLat[:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))
    dLon = np.copy(Lon)
    for la in range(Lat.shape[0]):
        dLon[la,:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))*np.cos(np.deg2rad(Lat[la,0]))
    Gridspacing = np.mean(np.append(dLat[:,:,None],dLon[:,:,None], axis=2))
    Area = dLat*dLon
    Area[Area < 0] = 0

    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    StartDay = Time[0]

    # connect over date line?
    if (Lon[0,0] < -176) & (Lon[0,-1] > 176):
        connectLon= 1
    else:
        connectLon= 0

    # ------------------------
    print('        track  precipitation')

    PRsmooth=gaussian_filter(DATA_all[:,:,:,Variables.index('PR')], sigma=(0,SmoothSigmaP,SmoothSigmaP))
    PRmask = (PRsmooth >= Pthreshold*dT)
    rgiObjectsPR, nr_objectsUD = ndimage.label(PRmask, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' precipitation object found')

    # connect objects over date line
    if connectLon == 1:
        rgiObjectsPR = ConnectLon(rgiObjectsPR)

    # remove None objects
    Objects=ndimage.find_objects(rgiObjectsPR)
    rgiVolObj=np.array([np.sum(rgiObjectsPR[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    ZERO_V =  np.where(rgiVolObj == 0)
    # Dummy = [slice(0, 1, None), slice(0, 1, None), slice(0, 1, None)]
    # Objects = np.array(Objects)
    # for jj in ZERO_V[0]:
    #     Objects[jj] = Dummy

    # Remove objects that are too small or short lived
    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1:]][rgiObjectsPR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsPR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])
    # create final object array
    PR_objects=np.copy(rgiObjectsPR); PR_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaPR*1000**2, np.ones(int(MinTimePR/dT)), mode='valid'))
        if (AreaTest == int(MinTimePR/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimePR/dT)):
            PR_objects[rgiObjectsPR == (ob+1)] = ii
            ii = ii + 1

    if connectLon == 1:
        print('        connect precipitation objects over date line')
        PR_objects = ConnectLon(PR_objects)

    grPRs = ObjectCharacteristics(PR_objects, # feature object file
                                DATA_all[:,:,:,Variables.index('PR')],         # original file used for feature detection
                                'PR_'+str(StartDay.year)+str(StartDay.month).zfill(2),
                                Time,            # timesteps of the data
                                Lat,             # 2D latidudes
                                Lon,             # 2D Longitudes
                                Gridspacing,
                                Area,
                                MinTime=int(MinTimePR/dT))      # minimum livetime in hours

    # --------------------------------------------------------
    print('        track  clouds')
    Csmooth=gaussian_filter(DATA_all[:,:,:,Variables.index('Tb')], sigma=(0,SmoothSigmaC,SmoothSigmaC))
    Cmask = (Csmooth <= Cthreshold)
    rgiObjectsC, nr_objectsUD = ndimage.label(Cmask, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' cloud object found')

    # connect objects over date line
    if connectLon == 1:
        rgiObjectsC = ConnectLon(rgiObjectsC)

    # minimum cloud volume
    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsC)

    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1:]][rgiObjectsC[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsC[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # rgiVolObjC=np.array([np.sum(rgiObjectsC[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])

    # create final object array
    C_objects=np.copy(rgiObjectsC); C_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaC*1000**2, np.ones(int(MinTimeC/dT)), mode='valid'))
        if (AreaTest == int(MinTimeC/dT)) & (len(rgiAreaObj[ob]) >=int(MinTimeC/dT)):
        # if rgiVolObjC[ob] >= MinAreaC:
            C_objects[rgiObjectsC == (ob+1)] = ii
            ii = ii + 1

    print('        break up long living cloud shield objects that heve many elements')
    C_objects = BreakupObjects(C_objects,
                                int(MinTimeC/dT),
                                dT)
    # connect objects over date line
    if connectLon == 1:
        print('        connect cloud objects over date line')
        C_objects = ConnectLon(C_objects)

    grCs = ObjectCharacteristics(C_objects, # feature object file
                                DATA_all[:,:,:,Variables.index('Tb')],         # original file used for feature detection
                                'Clouds_'+str(StartDay.year)+str(StartDay.month).zfill(2),
                                Time,            # timesteps of the data
                                Lat,             # 2D latidudes
                                Lon,             # 2D Longitudes
                                Gridspacing,
                                Area,
                                MinTime=int(MinTimeC/dT))      # minimum livetime in hours


    print('        check if pr objects quallify as MCS')
    # check if precipitation object is from an MCS
    Objects=ndimage.find_objects(PR_objects.astype(int))
    MCS_obj = np.copy(PR_objects); MCS_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = PR_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2:
            continue
        Cloud_ACT = np.copy(C_objects[Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        Area_ACT = Area[Objects[ii][1],Objects[ii][2]]
        PR_ACT = DATA_all[:,:,:,Variables.index('PR')][Objects[ii]]

        PR_Size = np.array([np.sum(Area_ACT[ObjACT[tt,:,:] >0]) for tt in range(ObjACT.shape[0])])
        PR_MAX = np.array([np.max(PR_ACT[tt,ObjACT[tt,:,:] >0]) if len(PR_ACT[tt,ObjACT[tt,:,:]>0]) > 0 else 0 for tt in range(ObjACT.shape[0])])
        # Get cloud shield
        rgiCL_obj = np.delete(np.unique(Cloud_ACT[ObjACT > 0]),0)
        if len(rgiCL_obj) == 0:
            # no deep cloud shield is over the precipitation
            continue
        CL_OB_TMP = C_objects[Objects[ii][0]]
        CL_TMP = DATA_all[:,:,:,Variables.index('Tb')][Objects[ii][0]]
        CLOUD_obj_act = np.in1d(CL_OB_TMP.flatten(), rgiCL_obj).reshape(CL_OB_TMP.shape)
        Cloud_Size = np.array([np.sum(Area[CLOUD_obj_act[tt,:,:] >0]) for tt in range(CLOUD_obj_act.shape[0])])
        # min temperatur must be taken over precip area
        CL_ob_pr = C_objects[Objects[ii]]
        CL_BT_pr = DATA_all[:,:,:,Variables.index('Tb')][Objects[ii]]
        Cloud_MinT = np.array([np.min(CL_BT_pr[tt,CL_ob_pr[tt,:,:] >0]) if len(CL_ob_pr[tt,CL_ob_pr[tt,:,:] >0]) > 0 else 0 for tt in range(CL_ob_pr.shape[0])])
        # minimum lifetime peak precipitation
        PR_MAXLT = np.copy(PR_MAX)
        PR_MAXLT[:] = np.max(PR_MAXLT)
        PR_MAXLT[:] = PR_MAXLT >= MCS_MinPeakPR*dT

        MCS_TEST = (Cloud_Size/1000**2 >= CL_Area) & (Cloud_MinT <= CL_MaxT) & (PR_Size/1000**2 >= MCS_Minsize) & (PR_MAX >= MCS_minPR*dT) & (PR_MAXLT == 1)

        # assign unique object numbers
        ObjACT = np.array(ObjACT).astype(int)
        ObjACT[ObjACT == 1] = ii+1

        # PR area defines MCS area and precipitation
        window_length = int(MCS_minTime/dT)
        cumulative_sum = np.cumsum(np.insert(MCS_TEST, 0, 0))
        moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
        if len(moving_averages) > 0:
            if np.max(moving_averages) == 1:
                TMP = np.copy(MCS_obj[Objects[ii]])
                TMP = TMP + ObjACT
                MCS_obj[Objects[ii]] = TMP
            else:
                continue
        else:
            continue

    rgiObjectsMCS, nr_objectsUD = ndimage.label(MCS_obj, structure=rgiObj_Struct)
    grMCSs = ObjectCharacteristics(rgiObjectsMCS, # feature object file
                                DATA_all[:,:,:,Variables.index('PR')],         # original file used for feature detection
                                'MCS_'+str(StartDay.year)+str(StartDay.month).zfill(2),
                                Time,            # timesteps of the data
                                Lat,             # 2D latidudes
                                Lon,             # 2D Longitudes
                                Gridspacing,
                                Area,
                                MinTime=int(MCS_minTime/dT))      # minimum livetime in hours

    print(' ')
    print('Save the object masks into a joint netCDF')
    # ============================
    # Write NetCDF
    iTime = np.array((Time - Time[0]).total_seconds()).astype('int')

    dataset = Dataset(NCfile,'w',format='NETCDF4_CLASSIC')
    yc = dataset.createDimension('yc', Lat.shape[0])
    xc = dataset.createDimension('xc', Lat.shape[1])
    time = dataset.createDimension('time', None)

    times = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('yc','xc',))
    lon = dataset.createVariable('lon', np.float32, ('yc','xc',))
    PR_real = dataset.createVariable('PR', np.float32,('time','yc','xc'))
    PR_obj = dataset.createVariable('PR_Objects', np.float32,('time','yc','xc'))
    MCSs = dataset.createVariable('MCS_Objects', np.float32,('time','yc','xc'))
    Cloud_real = dataset.createVariable('BT', np.float32,('time','yc','xc'))
    Cloud_obj = dataset.createVariable('BT_Objects', np.float32,('time','yc','xc'))

    times.calendar = "standard"
    times.units = "seconds since "+str(Time[0].year)+"-"+str(Time[0].month).zfill(2)+"-"+str(Time[0].day).zfill(2)+" "+str(Time[0].hour).zfill(2)+":"+str(Time[0].minute).zfill(2)+":00"
    times.standard_name = "time"
    times.long_name = "time"

    lat.long_name = "latitude" ;
    lat.units = "degrees_north" ;
    lat.standard_name = "latitude" ;

    lon.long_name = "longitude" ;
    lon.units = "degrees_east" ;
    lon.standard_name = "longitude" ;

    PR_real.coordinates = "lon lat"
    PR_obj.coordinates = "lon lat"
    MCSs.coordinates = "lon lat"
    Cloud_real.coordinates = "lon lat"
    Cloud_obj.coordinates = "lon lat"

    lat[:] = Lat
    lon[:] = Lon
    PR_real[:] = DATA_all[:,:,:,Variables.index('PR')]
    PR_obj[:] = PR_objects
    MCSs[:] = MCS_obj
    Cloud_real[:] = DATA_all[:,:,:,Variables.index('Tb')]
    Cloud_obj[:] = C_objects
    times[:] = iTime

    dataset.close()
    print('Saved: '+NCfile)

    return grMCSs, MCS_obj
