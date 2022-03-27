#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-03-27T16:02:35+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-03-27T16:02:48+02:00
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

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from scipy import ndimage
import xarray as xr

def calc_grid_distance_area(lat,lon):
    """Function to calculate grid distances and areas from coordinates
       This uses the spherical law of cosines
       lat,lon : input coordinates (degrees)
       dx : distance (m)
       dy : distance (m)
       gcellarea: area (m2)
       """

    earth_radius=6371000 #Earth radius in m

    dx = np.zeros(lat.shape)
    dy = np.zeros(lat.shape)
    gcellarea = np.zeros(lat.shape)

    lat = lat*np.pi/180.
    lon = lon*np.pi/180.

    dx[:,1:] = np.arccos(np.sin(lat[:,:-1])*np.sin(lat[:,1:]) + np.cos(lat[:,:-1])*np.cos(lat[:,1:])*np.cos(lon[:,1:]-lon[:,:-1]))*earth_radius

    dy[1:,:] = np.arccos(np.sin(lat[:-1,:])*np.sin(lat[1:,:]) + np.cos(lat[:-1,:])*np.cos(lat[1:,:])*np.cos(lon[1:,:]-lon[:-1,:]))*earth_radius

    #We approximate first column and row
    dx[:,0] = dx[:,1]
    dx[0,:] = dx[1,:]

    dy[:,0] = dy[:,1]
    dy[0,:] = dy[1,:]

    gcellarea = dx*dy

    return dx,dy,gcellarea

# def StormCharacteristics():

def check_PR_MCS(PRstorm_id, BTstorm_id, PRdata, BTdata, dt, Area, CL_Area, CL_MaxT, MCS_Minsize, MCS_minPR, MCS_MinPeakPR, MCS_minTime ):

    PRobj = ndimage.find_objects(PRstorm_id.astype(int))
    MCSobj = np.zeros(PRstorm_id.shape)

    for nst in range(len(PRobj)):

        if PRobj[nst]==None: continue

        PRobj_ACT = PRstorm_id[PRobj[nst]] == nst+1
        BTobj_ACT = BTstorm_id[PRobj[nst]]
        Area_ACT = Area[PRobj[nst][1],PRobj[nst][2]]
        PR_ACT = PRdata[PRobj[nst]]
        BT_ACT = BTdata[PRobj[nst]]

        PR_size = np.array([np.sum(Area_ACT[PRobj_ACT[tt,:,:]]) for tt in range(PRobj_ACT.shape[0])])
        PR_max  = np.array([np.max(PR_ACT[tt,PRobj_ACT[tt,:,:]]) for tt in range(PRobj_ACT.shape[0])])

        #Cloud shield

        PR_BT_ACT = np.delete(np.unique(BTobj_ACT[PRobj_ACT]),0)

        if len(PR_BT_ACT) == 0: continue # no deep cloud shield is over the precipitation

        #Select times of PR object only (all lat-lon)
        BTobj_TMP = BTstorm_id[PRobj[nst][0]]
        BT_TMP = BTdata[PRobj[nst][0]]

        cloudobj_ACT= np.in1d(BTobj_TMP.flatten(), PR_BT_ACT).reshape(BTobj_TMP.shape)

        cloud_size = np.array([np.sum(Area[cloudobj_ACT[tt,:,:]>0]) for tt in range(cloudobj_ACT.shape[0])])
        # min temperatur must be taken over precip area

        cloud_minT = np.array([np.min(BT_ACT[tt,BTobj_ACT[tt,:,:]>0]) if len(BTobj_ACT[tt,BTobj_ACT[tt,:,:]>0]) > 0 else 0 for tt in range(BTobj_ACT.shape[0])])
        #cloud_minT = np.array([np.min(CL_BT_pr[tt,CL_ob_pr[tt,:,:] >0]) if len(CL_ob_pr[tt,CL_ob_pr[tt,:,:] >0]) > 0 else 0 for tt in range(CL_ob_pr.shape[0])])


        MCS_TEST = (cloud_size/1000**2 >= CL_Area) &\
                   (cloud_minT <= CL_MaxT) &\
                   (PR_size >= MCS_Minsize) &\
                   (PR_max >= MCS_minPR) &\
                   (np.max(PR_max) >= MCS_MinPeakPR)

        PRobj_ACT = PRobj_ACT.astype(int)
        PRobj_ACT[PRobj_ACT == 1] = nst+1

        #PR area defines MCS area and precip

        window_length = int(MCS_minTime/dt)
        moving_averages = np.convolve(MCS_TEST, np.ones(window_length), 'valid') / window_length
        if len(moving_averages) > 0:
            if np.max(moving_averages) == 1:
                TMP = np.copy(MCSobj[PRobj[nst]])
                TMP = TMP + PRobj_ACT
                MCSobj[PRobj[nst]] = TMP
            else:
                continue
        else:
            continue

    return MCSobj

###########################################################
###########################################################

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



###########################################################
###########################################################
def MCStracking(PRdata,
                BTdata,
                time,
                Lon,
                Lat,
                dt,                        # time step of data in hours
                # Precipitation (PR) tracking setup
                SmoothSigmaPR = 0,          # Gaussion std for precipitation smoothing
                PRthreshold = 2,            # precipitation threshold [mm/h]
                MinTimePR = 3,             # minum lifetime of PR feature in hours
                MinAreaPR = 5000,          # minimum area of precipitation feature in km2
                # Brightness temperature (Tb) tracking setup
                SmoothSigmaBT = 0,          # Gaussion std for Tb smoothing
                BTthreshold = 241,          # minimum Tb of cloud shield
                MinTimeBT = 9,              # minium lifetime of cloud shield in hours
                MinAreaBT = 40000,          # minimum area of cloud shield in km2
                # MCs detection
                MCS_Minsize = 5000,        # km2
                MCS_minPR = 10,            # minimum max precipitation in mm/h
                MCS_MinPeakPR = 10,        # Minimum lifetime peak of MCS precipitation
                CL_MaxT = 225,             # minimum brightness temperature
                CL_Area = 40000,           # min cloud area size in km2
                MCS_minTime = 4,           # minimum time step
                NCfile = 'my20160701_CONUS-MCS-tracking.nc'):


    #@ Check that PR_data and BTdata have equal sizes


    if BTdata.shape != PRdata.shape:
        raise ValueError('Input arrays PR and BT must have same number of dimensions')


    ## If DX is not provided calculate it


    dx,dy,gcellarea=calc_grid_distance_area(Lat,Lon)


    Gridspacing = np.mean(np.append(dy[:,:,None],dx[:,:,None], axis=2))

    # crosses_dateline = False
    # if (Lon[0,0] < -176) & (Lon[0,-1] > 176): crosses_dateline = True

    ###########################################################
    ###########################################################

    print (' Tracking precipitation')

    PRsmooth=gaussian_filter(PRdata, sigma=(0,SmoothSigmaPR,SmoothSigmaPR))
    PRthres_mask = (PRsmooth>=PRthreshold)

    #PRthres_smooth = np.where(PRsmooth>=PRthreshold,PRsmooth,0)



    prObj_Struct = np.ones((3,3,3))
    storm_id,nstorms = label(PRthres_mask,structure=prObj_Struct)

    print (f'{nstorms} precipitation objects found')


    ###########################################################
    ###########################################################

    ## Connect over date line

    # if crosses_dateline:
    #     storm_id = connectLon(storm_id)

    ###########################################################
    ###########################################################

    new_id = 1
    PRstorm_id = np.zeros(storm_id.shape)
    if nstorms >=1:
        for nst in range(1,nstorms+1):

            start_time = time[0]
            PRstorm = PRdata.copy()
            this_storm = (storm_id == nst)
            time_storm = this_storm.any(axis=(1,2))
            valid_times = np.argwhere(time_storm).squeeze()

            if valid_times.size==1:
                valid_times = valid_times.reshape(1)

            PRstorm[~this_storm] = 0

            # 3D volumne time,lat,lon
            PRstorm_volume = np.sum(PRstorm)

            # Remove objects that are too small or short-lived
            # This condition means that it should be large enough over a number of consecutive timesteps
            # The default options mean the storm must be larger than 5000km2 during three hours

            PRstorm_area = np.array([np.sum(gcellarea[this_storm[tt,:,:]]) for tt in valid_times])

            area_test = np.max(np.convolve(PRstorm_area>=MinAreaPR*1000.**2,np.ones(int(MinTimePR/dt)), mode='valid'))

            if (area_test == int(MinTimePR/dt)):

                PRstorm_id [this_storm] = new_id
                new_id += 1

        # if crosses_dateline:
        #
        #     PRstorm_id = connectLon(PRstorm_id)

        #Calculate storm characteristics for selected storms

        # grPRs = StormCharacteristics(PRstorm_id,
        #                              PRdata,
        #                              f'PR_{time[valid_times[0].dt.strftime("%Y-%m-%d_%H:%M").item()}',
        #                              time,
        #                              Lat,
        #                              Lon,
        #                              Gridspacing,
        #                              gcellarea,
        #                              MinTime=int(MinTimePR/dt))

    ###########################################################
    ###########################################################

    print (' Tracking clouds') #Brightness temp

    BTsmooth=gaussian_filter(BTdata, sigma=(0,SmoothSigmaBT,SmoothSigmaBT))
    BTthres_mask = (BTsmooth <= BTthreshold)

    btObj_Struct = np.ones((3,3,3))
    storm_id,nstorms = label(BTthres_mask,structure=btObj_Struct)

    print (f'{nstorms} cloud objects found')

    ###########################################################
    ###########################################################

    ## Connect over date line

    # if crosses_dateline:
    #     storm_id = connectLon(storm_id)

    ###########################################################
    ###########################################################

    new_id = 1
    BTstorm_id = np.zeros(storm_id.shape)
    if nstorms >=1:
        for nst in range(1,nstorms+1):

            start_time = time[0]
            BTstorm = BTdata.copy()
            this_storm = (storm_id == nst)
            time_storm = this_storm.any(axis=(1,2))
            valid_times = np.argwhere(time_storm).squeeze()

            if valid_times.size==1:
                valid_times = valid_times.reshape(1)

            BTstorm[~this_storm] = 0


            # Remove objects that are too small or short-lived
            # This condition means that it should be large enough over a number of consecutive timesteps
            # The default options mean the storm must be larger than 5000km2 during three hours

            BTstorm_area = np.array([np.sum(gcellarea[this_storm[tt,:,:]]) for tt in valid_times])

            area_test = np.max(np.convolve(BTstorm_area>=MinAreaBT*1000.**2,np.ones(int(MinTimeBT/dt)), mode='valid'))

            if (area_test == int(MinTimeBT/dt)):

                BTstorm_id [this_storm] = new_id
                new_id += 1


        #Break up long living cloud shield objects that have many elements

        BTstorm_id = BreakupObjects(BTstorm_id.astype(int),
                                    int(MinTimeBT/dt),
                                    dt)

        # if crosses_dateline:
        #
        #     BTstorm_id = connectLon(BTstorm_id)
        #

        #Calculate storm characteristics for selected storms
        #
        # grBTs = StormCharacteristics(BTstorm_id,
        #                              BTdata,
        #                              f'BT_{time[valid_times[0].dt.strftime("%Y-%m-%d_%H:%M").item()}',
        #                              time,
        #                              Lat,
        #                              Lon,
        #                              Gridspacing,
        #                              gcellarea,
        #                              MinTime=int(MinTimeBT/dt))

    ###########################################################
    ###########################################################

    ## Check if PR storms qualify as MCS
    print ('Check if PR storm is a MCS')

    MCSobj = check_PR_MCS(PRstorm_id, BTstorm_id, PRdata, BTdata, dt, gcellarea, CL_Area, CL_MaxT, MCS_Minsize, MCS_minPR, MCS_MinPeakPR, MCS_minTime )


    MCSstorm_id,MCSnstorms = label(MCSobj,structure=np.ones((3,3,3)))


    #Calculate MCS storm characteristics for selected storms
    #
    # grMCSs = StormCharacteristics(MCSstorm_id,
    #                              PRdata,
    #                              f'MCS_{time[valid_times[0].dt.strftime("%Y-%m-%d_%H:%M").item()}',
    #                              time,
    #                              Lat,
    #                              Lon,
    #                              Gridspacing,
    #                              gcellarea,
    #                              MinTime=int(MCS_minTime/dt))

    ###########################################################
    ###########################################################

    print ('Save objects into a netCDF')


    fino=xr.Dataset({'MCS':(['time','y','x'],MCSstorm_id),
                     'PR_real':(['time','y','x'],PRdata),
                     'PR_obj':(['time','y','x'],PRstorm_id),
                     'BT_real':(['time','y','x'],BTdata),
                     'BT_obj':(['time','y','x'],BTstorm_id),
                     'lat':(['y','x'],Lat),
                     'lon':(['y','x'],Lon)},
                     coords={'time':time.values})

    fino.to_netcdf(NCfile,mode='w',encoding={'PR_real':{'zlib': True,'complevel': 5},
                                             'PR_obj':{'zlib': True,'complevel': 5},
                                             'BT_real':{'zlib': True,'complevel': 5},
                                             'BT_obj':{'zlib': True,'complevel': 5}})

    return 0, 0
