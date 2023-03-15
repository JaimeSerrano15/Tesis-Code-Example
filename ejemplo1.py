import numpy as np
import netCDF4 as nc
import salem
import xarray as xr
from shapely.wkt import loads as loadswkt
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd

# Plot Section 

shapefile = gpd.read_file('C:/Users/jaime/Documents/Workspace/Graduacion2023/Tesis/Codes/fields.shp')
#print(shapefile['geometry'])

# Print all the plots
fig, ax = plt.subplots(figsize=(10,10))
shapefile.plot(ax=ax)
#plt.show()

# Select an specific Plot
#print(shapefile.head())
plot_id = 1.0
plot = shapefile[shapefile['FID'] == plot_id]

fig, ax = plt.subplots(figsize=(10,10))
plot.plot(ax=ax)
#plt.show()

# Datacube Section
datacube = nc.Dataset('C:/Users/jaime/Documents/Workspace/Graduacion2023/Tesis/Codes/STAMARIA_S2_2023_cube.nc')

#print(datacube.variables['lon'][:])
#lon_dim = datacube.variables['B01']

xmin, ymin, xmax, ymax = plot.total_bounds

lon = datacube.variables['lon'][:]
lat = datacube.variables['lat'][:]

lon_indices = np.where((lon >= xmin) & (lon <= xmax))[0]
lat_indices = np.where((lat >= ymin) & (lat <= ymax))[0]

print("Lon: ", lon_indices)
print('Lat: ', lat_indices)

#print(datacube.variables['lon'][239])
#print(datacube.variables['lat'][371])

mask = []
for i in lon_indices:
    for j in lat_indices:
        mask.append((datacube.variables['lon'][i], datacube.variables['lat'][j]))
        
days = datacube['time']
ordered_idx = np.argsort(days).values
ordered_dates = days.values[ordered_idx]

print(ordered_dates)


