#!/usr/bin/env python

"""
   tracking_functions.py

   This file contains the tracking fuctions for the object
   identification and tracking of precipitation areas, cyclones,
   clouds, and moisture streams

"""
import glob
import os
from pdb import set_trace as stop
import pickle
from itertools import groupby
import datetime
import time
import logging


import numpy as np
import matplotlib.path as mplPath
import netCDF4 as nc
import pandas as pd
import xarray as xr

from scipy.ndimage import filters
from scipy.ndimage import morphology
from scipy import ndimage

from constants import const
import mcs_config as cfg


###########################################################
###########################################################

### UTILITY Functions
def calc_grid_distance_area(lat,lon):
    """ Function to calculate grid parameters
        It uses haversine function to approximate distances
        It approximates the first row and column to the sencond
        because coordinates of grid cell center are assumed
        lat, lon: input coordinates(degrees) 2D [y,x] dimensions
        dx: distance (m)
        dy: distance (m)
        area: area of grid cell (m2)
        grid_distance: average grid distance over the domain (m)
    """
    dy = np.zeros(lat.shape)
    dx = np.zeros(lon.shape)

    dx[:,1:]=haversine(lat[:,1:],lon[:,1:],lat[:,:-1],lon[:,:-1])
    dy[1:,:]=haversine(lat[1:,:],lon[1:,:],lat[:-1,:],lon[:-1,:])

    dx[:,0] = dx[:,1]
    dy[0,:] = dy[1,:]

    area = dx*dy
    grid_distance = np.mean(np.append(dy[:, :, None], dx[:, :, None], axis=2))

    return dx,dy,area,grid_distance

def haversine(lat1, lon1, lat2, lon2):

    """Function to calculate grid distances lat-lon
       This uses the Haversine formula
       lat,lon : input coordinates (degrees) - array or float
       dist_m : distance (m)
       https://en.wikipedia.org/wiki/Haversine_formula
       """
    # convert decimal degrees to radians
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + \
    np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    dist_m = c * const.earth_radius
    return dist_m

def calculate_area_objects(objects_id_pr,object_indices,grid_cell_area):

    """ Calculates the area of each object during their lifetime
        one area value for each object and each timestep it exist
    """
    num_objects = len(object_indices)
    area_objects = np.array(
        [
            [
            np.sum(grid_cell_area[object_indices[obj][1:]][objects_id_pr[object_indices[obj]][tstep, :, :] == obj + 1])
            for tstep in range(objects_id_pr[object_indices[obj]].shape[0])
            ]
        for obj in range(num_objects)
        ],
    dtype=object
    )

    return area_objects

def remove_small_short_objects(objects_id,area_objects,min_area,min_time,DT):
    """Checks if the object is large enough during enough time steps
        and removes objects that do not meet this condition
        area_object: array of lists with areas of each objects during their lifetime [objects[tsteps]]
        min_area: minimum area of the object (km2)
        min_time: minimum time with the object large enough (hours)
    """

    #create final object array
    sel_objects = np.zeros(objects_id.shape,dtype=int)

    new_obj_id = 1
    for obj,_ in enumerate(area_objects):
        AreaTest = np.max(
            np.convolve(
                np.array(area_objects[obj]) >= min_area * 1000**2,
                np.ones(int(min_time/ DT)),
                mode="valid",
            )
        )

        if (AreaTest == int(min_time/ DT)) & (
            len(area_objects[obj]) >= int(min_time/ DT)
        ):
            sel_objects[objects_id == (obj + 1)] =     new_obj_id
            new_obj_id += 1

    return sel_objects



###########################################################
###########################################################
def calc_object_characteristics(
    var_objects,  # feature object file
    var_data,  # original file used for feature detection
    filename_out,  # output file name and locaiton
    times,  # timesteps of the data
    Lat,  # 2D latidudes
    Lon,  # 2D Longitudes
    grid_spacing,  # average grid spacing
    grid_cell_area,
    min_tsteps=1  # minimum lifetime in data timesteps
    ):
    # ========

    num_objects = int(var_objects.max())
    object_indices = ndimage.find_objects(var_objects)

    if num_objects >= 1:
        objects_charac = {}
        logging.debug("            Loop over " + str(num_objects) + " objects")
        for iobj in range(num_objects):

            object_slice = np.copy(var_objects[object_indices[iobj]])
            data_slice   = np.copy(var_data[object_indices[iobj]])
            time_idx_slice = object_indices[iobj][0]
            lat_idx_slice  = object_indices[iobj][1]
            lon_idx_slice  = object_indices[iobj][2]

            #if len(object_slice) >= min_tsteps:
            if np.sum(np.any(object_slice==iobj+1,axis=(1,2))) >= min_tsteps:

                data_slice[object_slice!=(iobj + 1)] = np.nan
                grid_cell_area_slice = np.tile(grid_cell_area[lat_idx_slice, lon_idx_slice], (len(data_slice), 1, 1))
                grid_cell_area_slice[object_slice != (iobj + 1)] = np.nan
                lat_slice = Lat[lat_idx_slice, lon_idx_slice]
                lon_slice = Lon[lat_idx_slice, lon_idx_slice]


                # calculate statistics
                obj_times = times[time_idx_slice]
                obj_size  = np.nansum(grid_cell_area_slice, axis=(1, 2))
                obj_min = np.nanmin(data_slice, axis=(1, 2))
                obj_max = np.nanmax(data_slice, axis=(1, 2))
                obj_mean = np.nanmean(data_slice, axis=(1, 2))
                obj_tot = np.nansum(data_slice, axis=(1, 2))


                # Track lat/lon
                obj_mass_center = \
                np.array([ndimage.measurements.center_of_mass(object_slice[tt,:,:]==(iobj+1)) for tt in range(object_slice.shape[0])])

                if np.any(np.isnan(obj_mass_center)):
                    raise ValueError("mass center array contains NaNs")

                obj_track = np.full([len(obj_mass_center), 2], np.nan)

                obj_track[:,0]=np.array([lat_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center)])
                obj_track[:,1]=np.array([lon_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center)])

                if np.any(np.isnan(obj_track)):
                    raise ValueError("track array contains NaNs")

                obj_speed = (np.sum(np.diff(obj_mass_center,axis=0)**2,axis=1)**0.5) * (grid_spacing / 1000.0)

                this_object_charac = {
                    "mass_center_loc": obj_mass_center,
                    "speed": obj_speed,
                    "tot": obj_tot,
                    "min": obj_min,
                    "max": obj_max,
                    "mean": obj_mean,
                    "size": obj_size,
                    #                        'rgrAccumulation':rgrAccumulation,
                    "times": obj_times,
                    "track": obj_track,
                }

                try:
                    objects_charac[str(iobj + 1)] = this_object_charac
                except:
                    raise ValueError ("Error asigning properties to final dictionary")


        if filename_out is not None:
            with open(filename_out, 'wb') as handle:
                pickle.dump(objects_charac, handle)

        return objects_charac


# ==============================================================
# ==============================================================

def ConnectLon(object_indices):
    for tt in range(object_indices.shape[0]):
        EDGE = np.append(
            object_indices[tt, :, -1][:, None], object_indices[tt, :, 0][:, None], axis=1
        )
        iEDGE = np.sum(EDGE > 0, axis=1) == 2
        OBJ_Left = EDGE[iEDGE, 0]
        OBJ_Right = EDGE[iEDGE, 1]
        OBJ_joint = np.array(
            [
                OBJ_Left[ii].astype(str) + "_" + OBJ_Right[ii].astype(str)
                for ii,_ in enumerate(OBJ_Left)
            ]
        )
        NotSame = OBJ_Left != OBJ_Right
        OBJ_joint = OBJ_joint[NotSame]
        OBJ_unique = np.unique(OBJ_joint)
        # set the eastern object to the number of the western object in all timesteps
        for obj,_ in enumerate(OBJ_unique):
            ObE = int(OBJ_unique[obj].split("_")[1])
            ObW = int(OBJ_unique[obj].split("_")[0])
            object_indices[object_indices == ObE] = ObW
    return object_indices



### Break up long living cyclones by extracting the biggest cyclone at each time
def BreakupObjects(
    DATA,  # 3D matrix [time,lat,lon] containing the objects
    min_tsteps,  # minimum lifetime in data timesteps
    dT,
):  # time step in hours

    object_indices = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(24 / dT)  # min lifetime of object to be split
    AVmax = 1.5

    obj_structure_2D = np.zeros((3, 3, 3))
    obj_structure_2D[1, :, :] = 1
    rgiObjects2D, nr_objects2D = ndimage.label(DATA, structure=obj_structure_2D)

    rgiObjNrs = np.unique(DATA)[1:]
    TT = np.array([object_indices[obj][0].stop - object_indices[obj][0].start for obj in range(MaxOb)])
    # Sel_Obj = rgiObjNrs[TT > MinLif]

    # Average 2D objects in 3D objects?
    Av_2Dob = np.zeros((len(rgiObjNrs)))
    Av_2Dob[:] = np.nan
    ii = 1
    for obj,_ in enumerate(rgiObjNrs):
        #         if TT[obj] <= MinLif:
        #             # ignore short lived objects
        #             continue
        SelOb = rgiObjNrs[obj] - 1
        DATA_ACT = np.copy(DATA[object_indices[SelOb]])
        iOb = rgiObjNrs[obj]
        rgiObjects2D_ACT = np.copy(rgiObjects2D[object_indices[SelOb]])
        rgiObjects2D_ACT[DATA_ACT != iOb] = 0

        Av_2Dob[obj] = np.mean(
            np.array(
                [
                    len(np.unique(rgiObjects2D_ACT[tt, :, :])) - 1
                    for tt in range(DATA_ACT.shape[0])
                ]
            )
        )
        if Av_2Dob[obj] > AVmax:
            ObjectArray_ACT = np.copy(DATA_ACT)
            ObjectArray_ACT[:] = 0
            rgiObAct = np.unique(rgiObjects2D_ACT[0, :, :])[1:]
            for tt in range(1, rgiObjects2D_ACT[:, :, :].shape[0]):
                rgiObActCP = list(np.copy(rgiObAct))
                for ob1 in rgiObAct:
                    tt1_obj = list(
                        np.unique(
                            rgiObjects2D_ACT[tt, rgiObjects2D_ACT[tt - 1, :] == ob1]
                        )[1:]
                    )
                    if len(tt1_obj) == 0:
                        # this object ends here
                        rgiObActCP.remove(ob1)
                        continue
                    elif len(tt1_obj) == 1:
                        rgiObjects2D_ACT[
                            tt, rgiObjects2D_ACT[tt, :] == tt1_obj[0]
                        ] = ob1
                    else:
                        VOL = [
                            np.sum(rgiObjects2D_ACT[tt, :] == tt1_obj[jj])
                            for jj,_ in enumerate(tt1_obj)
                        ]
                        rgiObjects2D_ACT[
                            tt, rgiObjects2D_ACT[tt, :] == tt1_obj[np.argmax(VOL)]
                        ] = ob1
                        tt1_obj.remove(tt1_obj[np.argmax(VOL)])
                        rgiObActCP = rgiObActCP + list(tt1_obj)

                # make sure that mergers are assigned the largest object
                for ob2 in rgiObActCP:
                    ttm1_obj = list(
                        np.unique(
                            rgiObjects2D_ACT[tt - 1, rgiObjects2D_ACT[tt, :] == ob2]
                        )[1:]
                    )
                    if len(ttm1_obj) > 1:
                        VOL = [
                            np.sum(rgiObjects2D_ACT[tt - 1, :] == ttm1_obj[jj])
                            for jj,_ in enumerate(ttm1_obj)
                        ]
                        rgiObjects2D_ACT[tt, rgiObjects2D_ACT[tt, :] == ob2] = ttm1_obj[
                            np.argmax(VOL)
                        ]

                # are there new object?
                NewObj = np.unique(rgiObjects2D_ACT[tt, :, :])[1:]
                NewObj = list(np.setdiff1d(NewObj, rgiObAct))
                if len(NewObj) != 0:
                    rgiObActCP = rgiObActCP + NewObj
                rgiObActCP = np.unique(rgiObActCP)
                rgiObAct = np.copy(rgiObActCP)

            rgiObjects2D_ACT[rgiObjects2D_ACT != 0] = np.copy(
                rgiObjects2D_ACT[rgiObjects2D_ACT != 0] + MaxOb
            )
            MaxOb = np.max(DATA)

            # save the new objects to the original object array
            TMP = np.copy(DATA[object_indices[SelOb]])
            TMP[rgiObjects2D_ACT != 0] = rgiObjects2D_ACT[rgiObjects2D_ACT != 0]
            DATA[object_indices[SelOb]] = np.copy(TMP)

    # clean up object matrix
    Unique = np.unique(DATA)[1:]
    object_indices = ndimage.find_objects(DATA)
    rgiVolObj = np.array(
        [
            np.sum(DATA[object_indices[Unique[obj] - 1]] == Unique[obj])
            for obj,_ in enumerate(Unique)
        ]
    )
    TT = np.array(
        [
            object_indices[Unique[obj] - 1][0].stop - object_indices[Unique[obj] - 1][0].start
            for obj,_ in enumerate(Unique)
        ]
    )

    # create final object array
    CY_objectsTMP = np.copy(DATA)
    CY_objectsTMP[:] = 0
    ii = 1
    for obj,_ in enumerate(rgiVolObj):
        if TT[obj] >= min_tsteps / dT:
            CY_objectsTMP[DATA == Unique[obj]] = ii
            ii = ii + 1

    # lable the objects from 1 to N
    DATA_fin = np.copy(CY_objectsTMP)
    DATA_fin[:] = 0
    Unique = np.unique(CY_objectsTMP)[1:]
    ii = 1
    for obj,_ in enumerate(Unique):
        DATA_fin[CY_objectsTMP == Unique[obj]] = ii
        ii = ii + 1

    return DATA_fin

############################################################
###########################################################
#### ======================================================
# function to perform MCS tracking
def MCStracking(
    pr_data,
    bt_data,
    times,
    Lon,
    Lat,
    nc_file
):
    """ Function to track MCS from precipitation and brightness temperature
    """

    start_time = time.time()
    #Reading tracking parameters

    DT = cfg.DT

    #Precipitation tracking setup
    smooth_sigma_pr = cfg.smooth_sigma_pr   # [0] Gaussion std for precipitation smoothing
    thres_pr        = cfg.thres_pr     # [2] precipitation threshold [mm/h]
    min_time_pr     = cfg.min_time_pr     # [3] minum lifetime of PR feature in hours
    min_area_pr     = cfg.min_area_pr      # [5000] minimum area of precipitation feature in km2
    # Brightness temperature (Tb) tracking setup
    smooth_sigma_bt = cfg.smooth_sigma_bt   #  [0] Gaussion std for Tb smoothing
    thres_bt        = cfg.thres_bt     # [241] minimum Tb of cloud shield
    min_time_bt     = cfg.min_time_bt       # [9] minium lifetime of cloud shield in hours
    min_area_bt     = cfg.min_area_bt       # [40000] minimum area of cloud shield in km2
    # MCs detection
    MCS_min_area      = cfg.MCS_min_area    # [5000] km2
    MCS_thres_pr       = cfg.MCS_thres_pr      # [10] minimum max precipitation in mm/h
    MCS_thres_peak_pr   = cfg.MCS_thres_peak_pr  # [10] Minimum lifetime peak of MCS precipitation
    MCS_thres_bt     = cfg.MCS_thres_bt        # [225] minimum brightness temperature
    MCS_min_area_bt         = cfg.MCS_min_area_bt        # [40000] min cloud area size in km2
    MCS_min_time     = cfg.MCS_min_time    # [4] minimum time step

    #Calculating grid distances and areas

    _,_,grid_cell_area,grid_spacing = calc_grid_distance_area(Lat,Lon)
    grid_cell_area[grid_cell_area < 0] = 0

    obj_structure_3D = np.ones((3,3,3))

    start_day = times[0]


    # connect over date line?
    crosses_dateline = False
    if (Lon[0, 0] < -176) & (Lon[0, -1] > 176):
        crosses_dateline = True

    end_time = time.time()
    logging.debug(f"======> 'Initialize MCS tracking function: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # TRACKING PRECIP OBJECTS
    # --------------------------------------------------------
    logging.debug("        track  precipitation")

    pr_smooth= filters.gaussian_filter(
        pr_data, sigma=(0, smooth_sigma_pr, smooth_sigma_pr)
    )
    pr_mask = pr_smooth >= thres_pr * DT
    objects_id_pr, num_objects = ndimage.label(pr_mask, structure=obj_structure_3D)
    logging.debug("            " + str(num_objects) + " precipitation object found")

    # connect objects over date line
    if crosses_dateline:
        objects_id_pr = ConnectLon(objects_id_pr)

    # get indices of object to reduce memory requirements during manipulation
    object_indices = ndimage.find_objects(objects_id_pr)


    #Calcualte area of objects
    area_objects = calculate_area_objects(objects_id_pr,object_indices,grid_cell_area)

    # Keep only large and long enough objects
    # Remove objects that are too small or short lived
    pr_objects = remove_small_short_objects(objects_id_pr,area_objects,min_area_pr,min_time_pr,DT)

    grPRs = calc_object_characteristics(
        pr_objects,  # feature object file
        pr_data,  # original file used for feature detection
        f"{cfg.path_in}/PR_{start_day.year}{start_day.month:02d}",
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(min_time_pr/ DT), # minimum lifetime in data timesteps
    )

    end_time = time.time()
    logging.debug(f"======> 'Tracking precip: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # TRACKING CLOUD (BT) OBJECTS
    # --------------------------------------------------------
    logging.debug("            track  clouds")
    bt_smooth = filters.gaussian_filter(
        bt_data, sigma=(0, smooth_sigma_bt, smooth_sigma_bt)
    )
    bt_mask = bt_smooth <= thres_bt
    objects_id_bt, num_objects = ndimage.label(bt_mask, structure=obj_structure_3D)
    logging.debug("            " + str(num_objects) + " cloud object found")

    # connect objects over date line
    if crosses_dateline:
        logging.debug("            connect cloud objects over date line")
        objects_id_bt = ConnectLon(objects_id_bt)

    # get indices of object to reduce memory requirements during manipulation
    object_indices = ndimage.find_objects(objects_id_bt)

    #Calcualte area of objects
    area_objects = calculate_area_objects(objects_id_bt,object_indices,grid_cell_area)

    # Keep only large and long enough objects
    # Remove objects that are too small or short lived
    bt_objects = remove_small_short_objects(objects_id_bt,area_objects,min_area_bt,min_time_bt,DT)

    end_time = time.time()
    logging.debug(f"======> 'Tracking clouds: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    #logging.debug("            break up long living cloud shield objects that heve many elements")
    #bt_objects = BreakupObjects(bt_objects, int(min_time_bt / DT), DT)

    end_time = time.time()
    logging.debug(f"======> 'Breaking up cloud objects: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    grCs = calc_object_characteristics(
        bt_objects,  # feature object file
        bt_data,  # original file used for feature detection
        f"{cfg.path_in}/BT_{start_day.year}{start_day.month:02d}",
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(min_time_bt / DT), # minimum lifetime in data timesteps
    )
    end_time = time.time()
    logging.debug(f"======> 'Calculate cloud characteristics: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # CHECK IF PR OBJECTS QUALIFY AS MCS
    # (or selected strom type according to msc_config.py)
    # --------------------------------------------------------
    logging.debug("            check if pr objects quallify as MCS (or selected storm type)")
    # check if precipitation object is from an MCS
    object_indices = ndimage.find_objects(pr_objects)
    MCS_objects = np.zeros(pr_objects.shape,dtype=int)

    for iobj,_ in enumerate(object_indices):

        if object_indices[iobj] is None:
            continue

        time_slice = object_indices[iobj][0]
        lat_slice  = object_indices[iobj][1]
        lon_slice  = object_indices[iobj][2]


        pr_object_slice= pr_objects[object_indices[iobj]]
        pr_object_act = np.where(pr_object_slice==iobj+1,True,False)

        if len(pr_object_act) < 2:
            continue

        pr_slice =  pr_data[object_indices[iobj]]
        pr_act = np.copy(pr_slice)
        pr_act[~pr_object_act] = 0

        bt_slice  = bt_data[object_indices[iobj]]
        bt_act = np.copy(bt_slice)
        bt_act[~pr_object_act] = 0

        bt_object_slice = bt_objects[object_indices[iobj]]
        bt_object_act = np.copy(bt_object_slice)
        bt_object_act[~pr_object_act] = 0

        area_act = np.tile(grid_cell_area[lat_slice, lon_slice], (pr_act.shape[0], 1, 1))
        area_act[~pr_object_act] = 0



        pr_size = np.array(np.sum(area_act,axis=(1,2)))
        pr_max = np.array(np.max(pr_act,axis=(1,2)))


        #Check overlaps between clouds (bt) and precip objects
        objects_overlap = np.delete(np.unique(bt_object_act[pr_object_act]),0)

        if len(objects_overlap) == 0:
            # no deep cloud shield is over the precipitation
            continue

        ## Keep bt objects (entire) that partially overlap with pr object

        bt_object_overlap = np.in1d(bt_objects[time_slice].flatten(), objects_overlap).reshape(bt_objects[time_slice].shape)

        # Get size of all cloud (bt) objects together
        # We get size of all cloud objects that overlap partially with pr object
        # DO WE REALLY NEED THIS?

        bt_size = np.array(
            [
            np.sum(grid_cell_area[bt_object_overlap[tt, :, :] > 0])
            for tt in range(bt_object_overlap.shape[0])
            ]
        )

        #Check if BT is below threshold over precip areas
        bt_min_temp = np.nanmin(np.where(bt_object_slice>0,bt_slice,999),axis=(1,2))



        # minimum lifetime peak precipitation
        is_pr_peak_intense = np.max(pr_max) >= MCS_thres_peak_pr * DT
        MCS_test = (
            (bt_size / 1000**2 >= MCS_min_area_bt)
            & (bt_min_temp  <= MCS_thres_bt )
            & (pr_size / 1000**2 >= MCS_min_area )
            & (pr_max >= MCS_thres_pr * DT)
            & (is_pr_peak_intense)
        )

        # assign unique object numbers

        pr_object_act = np.array(pr_object_act).astype(int)
        pr_object_act[pr_object_act == 1] = iobj + 1

        window_length = int(MCS_min_time / DT)
        moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length
        if (len(moving_averages) > 0) & (np.max(moving_averages) == 1):
            TMP = np.copy(MCS_objects[object_indices[iobj]])
            TMP = TMP + pr_object_act
            MCS_objects[object_indices[iobj]] = TMP

        else:
            continue

    #if len(objects_overlap)>1: import pdb; pdb.set_trace()
    objects_id_MCS, num_objects = ndimage.label(MCS_objects, structure=obj_structure_3D)
    grMCSs = calc_object_characteristics(
        objects_id_MCS,  # feature object file
        pr_data,  # original file used for feature detection
        f"{cfg.path_in}/MCS_{start_day.year}{start_day.month:02d}",
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(MCS_min_time / DT), # minimum lifetime in data timesteps
    )

    end_time = time.time()
    logging.debug(f"======> 'MCS tracking: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()


    ###########################################################
    ###########################################################
    ## WRite netCDF with xarray
    if nc_file is not None:
        logging.debug ('Save objects into a netCDF')

        fino=xr.Dataset({'MCS_objects':(['time','y','x'],objects_id_MCS),
                         'PR':(['time','y','x'],pr_data),
                         'PR_objects':(['time','y','x'],objects_id_pr),
                         'BT':(['time','y','x'],bt_data),
                         'BT_objects':(['time','y','x'],objects_id_bt),
                         'lat':(['y','x'],Lat),
                         'lon':(['y','x'],Lon)},
                         coords={'time':times.values})

        fino.to_netcdf(nc_file,mode='w',encoding={'PR':{'zlib': True,'complevel': 5},
                                                 'PR_objects':{'zlib': True,'complevel': 5},
                                                 'BT':{'zlib': True,'complevel': 5},
                                                 'BT_objects':{'zlib': True,'complevel': 5},
                                                 'MCS_objects':{'zlib': True,'complevel': 5}})


    # fino = xr.Dataset({
    # 'MCS_objects': xr.DataArray(
    #             data   = objects_id_MCS,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Mesoscale Convective System objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'PR_objects': xr.DataArray(
    #             data   = objects_id_pr,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Precipitation objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'BT_objects': xr.DataArray(
    #             data   = objects_id_bt,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Cloud (brightness temperature) objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'PR': xr.DataArray(
    #             data   = pr_data,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Precipitation',
    #                 'standard_name': 'precipitation',
    #                 'units'     : 'mm h-1',
    #                 }
    #             ),
    # 'BT': xr.DataArray(
    #             data   = bt_data,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Brightness temperature',
    #                 'standard_name': 'brightness_temperature',
    #                 'units'     : 'K',
    #                 }
    #             ),
    # 'lat': xr.DataArray(
    #             data   = Lat,   # enter data here
    #             dims   = ['y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': "latitude",
    #                 'standard_name': "latitude",
    #                 'units'     : "degrees_north",
    #                 }
    #             ),
    # 'lon': xr.DataArray(
    #             data   = Lon,   # enter data here
    #             dims   = ['y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': "longitude",
    #                 'standard_name': "longitude",
    #                 'units'     : "degrees_east",
    #                 }
    #             ),
    #         },
    #     attrs = {'date':datetime.date.today().strftime('%Y-%m-%d'),
    #              "comments": "File created with MCS_tracking"},
    #     coords={'time':times.values}
    # )


    # fino.to_netcdf(nc_file,mode='w',format = "NETCDF4",
    #                encoding={'PR':{'zlib': True,'complevel': 5},
    #                          'PR_objects':{'zlib': True,'complevel': 5},
    #                          'BT':{'zlib': True,'complevel': 5},
    #                          'BT_objects':{'zlib': True,'complevel': 5}})


        end_time = time.time()
        logging.debug(f"======> 'Writing files: {(end_time-start_time):.2f} seconds \n")
        start_time = time.time()
    else:
        logging.debug(f"No writing files required, output file name is empty")
    ###########################################################
    ###########################################################
    # ============================
    # Write NetCDF
    return grMCSs, MCS_objects
