import numpy as np
import netCDF4
import salem
import xarray as xr
from shapely.wkt import loads as loadswkt
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd

####################################
# Reading the Plots and Getting a single Plot
shapefile = gpd.read_file('C:/Users/jaime/Documents/Workspace/Graduacion2023/Tesis/Codes/fields.shp')

fig, ax = plt.subplots(figsize=(10,10))
shapefile.plot(ax=ax)
plt.show()

plot_id = 1.0
plot = shapefile[shapefile['FID'] == plot_id]

fig, ax = plt.subplots(figsize=(10,10))
plot.plot(ax=ax)
plt.show()



###############################################################
# Getting Datacube
ncfile = 'C:/Users/jaime/Documents/Workspace/Graduacion2023/Tesis/Codes/STAMARIA_S2_2023_cube.nc'
ds = salem.open_xr_dataset(ncfile)

days = ds['time']
ordered_id = np.argsort(days).values
ordered_days = days.values[ordered_id]


##############################################################
# Finding the Datacube pixels that are within the plot

lon_min, lat_min, lon_max, lat_max = plot.total_bounds

filter_ds = ds.where((ds['lon']>=lon_min) & (ds['lon'] <= lon_max) & (ds['lat'] >= lat_min) & (ds['lat'] <= lat_max))


##############################################################
# Visualizing Band Data

second_date = ordered_days[2]
Band12_date0 = ds['B12'].sel(time=second_date)
Band12_date0_filter = filter_ds['B12'].sel(time=second_date)

matplotlib.rcParams['figure.figsize'] = [20,10]

vmin = np.nanpercentile(Band12_date0.values, 2)
vmax = np.nanpercentile(Band12_date0.values, 98)
plt.imshow(Band12_date0, cmap='Spectral', vmin=vmin, vmax=vmax)

plt.show()

vmin = np.nanpercentile(Band12_date0_filter.values, 2)
vmax = np.nanpercentile(Band12_date0_filter.values, 98)
plt.imshow(Band12_date0_filter, cmap='Spectral', vmin=vmin, vmax=vmax)

plt.show()

##################################################################
# Creating new Variables
ds = ds.assign(NDVI= lambda x: (x.B08-x.B04)/(x.B08+x.B04))
filter_ds = filter_ds.assign(NDVI= lambda x: (x.B08-x.B04)/(x.B08+x.B04))

NDVI_date2 = ds['NDVI'].sel(time=second_date)

matplotlib.rcParams['figure.figsize'] = [20,10]

vmin = np.nanpercentile(NDVI_date2.values, 2)
vmax = np.nanpercentile(NDVI_date2.values, 98)
plt.imshow(NDVI_date2, cmap='viridis', vmin=vmin, vmax=vmax)
plt.show()

NDVI_date2 = filter_ds['NDVI'].sel(time=second_date)

matplotlib.rcParams['figure.figsize'] = [20,10]

vmin = np.nanpercentile(NDVI_date2.values, 2)
vmax = np.nanpercentile(NDVI_date2.values, 98)
plt.imshow(NDVI_date2, cmap='viridis', vmin=vmin, vmax=vmax)
plt.show()

###############################################################
# Calculating the mean value fo the NDVI
mean_ndvi = []


for date in ordered_days:
    mean_ndvi.append(np.mean(filter_ds['NDVI'].sel(time=date)))

fig = plt.figure(figsize=(10,5))

plt.bar(ordered_days,mean_ndvi, color='green', width=0.2)
plt.title("NDVI avg per date")
plt.show()