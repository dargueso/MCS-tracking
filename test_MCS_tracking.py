import pytest
import xarray as xr
import numpy as np
import pandas as pd


from constants import const
from tracking_functions_optimized import calc_grid_distance_area
from tracking_functions_optimized import MCStracking

def test_area_simple():

    lat = np.array([[39.0,40.0],[39.1,40.1]])
    lon = np.array([[1.0,2.0],[1.1,2.1]])
    dx,dy,area,grid_spacing = calc_grid_distance_area(lat,lon)
    assert area[0,1] == 1966730866.401264

def test_area():

    pr_test = xr.open_dataset("./RAIN_test.nc")
    dx, dy, area, grid_spacing = calc_grid_distance_area(pr_test.lat.values,pr_test.lon.values)
    assert area[100,100] == 4174262.3368386314
    assert dx[100,100] == 2043.120533315232
    assert dy[100,100] == 2043.0817804299295
    assert grid_spacing == 2061.1917602097433

def test_MCStracking():

    pr_test = xr.open_dataset("./RAIN_test.nc")
    olr_test = xr.open_dataset("./OLR_test.nc")

    pr_data = pr_test.RAIN.values
    bt_data = (olr_test.OLR.squeeze().values / const.SB_sigma) ** (0.25)

    lat = pr_test.lat.values
    lon = pr_test.lon.values

    times = pd.date_range(pr_test.time.isel(time=0).values, end=pr_test.time.isel(time=-1).values, freq='1H')

    grMCSs, MCS_objects = MCStracking(
            pr_data,
            bt_data,
            times,
            lon,
            lat,
            nc_file          =   None,
        )


    assert MCS_objects.max() == 17
    assert float(grMCSs['1']['rgrPR_Vol'].max())== 17591.9140625
    assert float(grMCSs['7']['rgrPR_Vol'][4]) == 9674.66015625
    assert float(grMCSs['2']['rgrPR_Max'][-1]) == 10.741204261779785
