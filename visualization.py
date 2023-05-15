import xarray as xr
import geopandas as gpd

# Open the NetCDF file
ncfile = xr.open_dataset('Data/sta_maria/STAMARIA_S2_2023_cube.nc')
unique_dates = ncfile['time'].values
print(unique_dates)

# Print the structure of the NetCDF cube
#print(ncfile)

shapefile_path = 'Data/sta_maria/fields.shp'
df = gpd.read_file(shapefile_path)

# Droping rows with no geometry
df = df.dropna(subset=['geometry'])
nan_geometry_rows = df.loc[df['geometry'].isna()]
#print(nan_geometry_rows)

# Print header
#print(list(df.columns))

# Print values for first 10 rows
#for index, row in df.iterrows():
#    print("Row", index)
#    print(row)


