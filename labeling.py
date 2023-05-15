import xarray as xr
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio import Affine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from skimage.measure import grid_points_in_poly

datacube = xr.open_dataset("Data/sta_maria/STAMARIA_S2_2023_cube.nc")
shapefile = gpd.read_file("Data/sta_maria/fields.shp")

shapefile = shapefile.dropna(subset=['geometry'])

def calculate_transform(datacube):
    lat_res = (datacube.lat[-1] - datacube.lat[0]) / (len(datacube.lat) - 1)
    lon_res = (datacube.lon[-1] - datacube.lon[0]) / (len(datacube.lon) - 1)
    return Affine(lon_res, 0, datacube.lon[0], 0, -lat_res, datacube.lat[-1])

transform = calculate_transform(datacube)

def create_labeled_structure(datacube, shapefile, transform):
    labeled_structure = np.zeros((len(datacube['lat']), len(datacube['lon'])), dtype=int) - 1
    for idx, row in shapefile.iterrows():
        mask = geometry_mask([row['geometry']],
                             transform=transform,
                             out_shape=(len(datacube['lat']), len(datacube['lon'])),
                             invert=True)
        labeled_structure[mask] = idx
    return labeled_structure

labeled_structure = create_labeled_structure(datacube, shapefile, transform)
print(labeled_structure)

def prepare_data_for_cnn(datacube, labeled_structure):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    x_train = np.stack([datacube[band].values for band in bands], axis=-1)
    y_train = labeled_structure
    return x_train, y_train

def visualize_labeled_structure(labeled_structure, shapefile):
    unique_values = np.unique(labeled_structure)
    unique_values = unique_values[unique_values != 0]  # exclude background value (0)
    cmap = plt.cm.get_cmap('tab20', len(unique_values))  # choose a colormap with enough colors
    plt.figure(figsize=(10, 10))
    plt.imshow(labeled_structure, cmap=cmap, origin='upper')

    # Create legend
    handles = []
    for plot_id in unique_values:
        label = f'Plot {plot_id}'
        color = cmap(plot_id / len(unique_values))
        handles.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

visualize_labeled_structure(labeled_structure, shapefile)

#x_train, y_train = prepare_data_for_cnn(datacube, labeled_structure)

"""
with rasterio.open(
    "labeled_structure.tif",
    'w',
    driver='GTiff',
    height=labeled_structure.shape[0],
    width=labeled_structure.shape[1],
    count=1,
    dtype=labeled_structure.dtype,
    crs=shapefile.crs,
    transform=transform
) as dst:
    dst.write(labeled_structure, 1)
"""


