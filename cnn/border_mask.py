import argparse
#import netCDF4 as nc
import salem
import datetime
import numpy as np
import geopandas as gpd
import xarray as xr
import time
import matplotlib.pyplot as plt
from rasterio import features
from scipy.signal import convolve2d

datacube = xr.open_dataset("data/sta_maria/STAMARIA_S2_2023_cube.nc", chunks=150) # Cambiar rutas a donde tengas tu data guardada
shapefile = gpd.read_file("data/sta_maria/fields.shp") # Cambiar rutas a donde tengas tu data guardada
shapefile = shapefile.dropna(subset=['geometry'])

test_field = shapefile.geometry[1054]
print(shapefile)

datacube = datacube.assign(ONES=lambda x: 1 * np.isnan((x.B02 * np.nan)))
print(datacube)

all_boundaries = np.full((datacube.dims['lat'], datacube.dims['lon']), False)

for field_geo in shapefile.geometry:
    datacube_roi = datacube.salem.roi(geometry=field_geo, other=0.0)

    o_fields_zero_one = datacube_roi['ONES'][0, :]

    conv_1_kernel = np.ones((3, 3))
    big_field = convolve2d(o_fields_zero_one, conv_1_kernel, 'same')

    big_field_zero_one = (big_field > 0) * 1.0

    field_boundary = (big_field_zero_one - o_fields_zero_one) == 1

    all_boundaries = np.logical_or(all_boundaries, field_boundary)


plt.imshow(all_boundaries)
plt.show()
