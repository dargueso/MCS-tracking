import pytest
import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage

from mcstracking import const
from mcstracking import calc_grid_distance_area
from mcstracking import calculate_area_objects
from mcstracking import remove_small_short_objects
from mcstracking import MCStracking
from mcstracking import default_config

from conftest import DATA, needs_data

def test_area_simple():

    lat = np.array([[39.0,40.0],[39.1,40.1]])
    lon = np.array([[1.0,2.0],[1.1,2.1]])
    dx,dy,area,grid_spacing = calc_grid_distance_area(lat,lon)
    assert np.isclose(area[0,1], 1966730866.401264, rtol=1e-12)

@needs_data
def test_area():

    pr_test = xr.open_dataset(f"{DATA}/RAIN_test.nc")
    dx, dy, area, grid_spacing = calc_grid_distance_area(pr_test.lat.values,pr_test.lon.values)
    # float64 values (the file holds float32 coordinates; the tracker promotes
    # them). Before v2.0 (re-cut) the area was computed in float32 and read
    # 4174262.3368 here, depending on the numpy version.
    assert np.isclose(area[100,100], 4173593.0900924527, rtol=1e-12)
    assert np.isclose(dx[100,100], 2043.0957234457928, rtol=1e-12)
    assert np.isclose(dy[100,100], 2042.7790250832984, rtol=1e-12)
    assert np.isclose(grid_spacing, 2061.1917937651656, rtol=1e-12)

@needs_data
def test_MCStracking(tmp_path):

    pr_test = xr.open_dataset(f"{DATA}/RAIN_test.nc")
    olr_test = xr.open_dataset(f"{DATA}/OLR_test.nc")

    pr_data = pr_test.RAIN.values
    bt_data = (olr_test.OLR.squeeze().values / const.SB_sigma) ** (0.25)

    lat = pr_test.lat.values
    lon = pr_test.lon.values

    times = pd.date_range(pr_test.time.isel(time=0).values, end=pr_test.time.isel(time=-1).values, freq='1h')

    grMCSs, MCS_objects = MCStracking(
            pr_data,
            bt_data,
            times,
            lon,
            lat,
            nc_file          =   str(tmp_path / "test.nc"),
            path_out         =   str(tmp_path),
            cfg              =   default_config,
        )


    assert MCS_objects.max() == 17
    assert float(grMCSs['1']['tot'].max())== 17591.9140625
    assert float(grMCSs['6']['tot'][4]) == 6056.83984375
    assert float(grMCSs['2']['max'][-1]) == 10.741204261779785


def test_area_obj():

    test_obj_id=np.zeros((10,10,10),dtype=int)
    test_obj_id[2:5,4:9,5:8]=1
    test_obj_id[6:8,4:8,5:7]=2

    test_object_indices = ndimage.find_objects(test_obj_id)

    area_objects = calculate_area_objects(test_obj_id,test_object_indices,np.ones((10,10),dtype=float))

    assert len(area_objects) == 2
    assert area_objects[0] == [15.0,15.0,15.0]
    assert area_objects[1] == [8.0,8.0]

def test_remove_small_short_objects():

    test_obj_id=np.zeros((10,10,10),dtype=int)
    test_obj_id[2:5,4:9,5:8]=1
    test_obj_id[6:8,4:8,5:7]=2
    test_obj_id[5:6,3:9,3:7]=2

    #Using a dx and dy of 2000m
    #The areas are

    test_object_indices = ndimage.find_objects(test_obj_id)
    area_objects = np.array([[60000000,60000000,60000000],[24000000,32000000,32000000]], dtype=object)

    large_obj_id = remove_small_short_objects(test_obj_id,area_objects,30,3,1)


    assert int(large_obj_id[large_obj_id==1].sum()) == 45
    assert int(large_obj_id[large_obj_id==2].sum()) == 0
