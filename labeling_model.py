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
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate

datacube = xr.open_dataset("Data/sta_maria/STAMARIA_S2_2023_cube.nc")
shapefile = gpd.read_file("Data/sta_maria/fields.shp")

shapefile = shapefile.dropna(subset=['geometry'])

shapefile['plot_id'] = range(1, len(shapefile) + 1)

def calculate_transform(datacube):
    lat_res = (datacube.lat[-1] - datacube.lat[0]) / (len(datacube.lat) - 1)
    lon_res = (datacube.lon[-1] - datacube.lon[0]) / (len(datacube.lon) - 1)
    return Affine(lon_res, 0, datacube.lon[0], 0, -lat_res, datacube.lat[-1])

transform = calculate_transform(datacube)

def create_labeled_structure(datacube, shapefile, transform):
    labeled_structure = np.zeros((len(datacube['lat']), len(datacube['lon'])), dtype=int)
    for _, row in shapefile.iterrows():
        mask = geometry_mask([row['geometry']],
                             transform=transform,
                             out_shape=(len(datacube['lat']), len(datacube['lon'])),
                             invert=True)
        labeled_structure[mask] = row['plot_id']
    return labeled_structure

labeled_structure = create_labeled_structure(datacube, shapefile, transform)

def create_cnn(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up1 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2), conv1], axis=3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)

    up2 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv4)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def preprocess_data(x, y, test_size=0.2, val_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test


x = np.expand_dims(labeled_structure, axis=0)
y = np.expand_dims(labeled_structure, axis=0)

x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(x, y)

num_classes = len(np.unique(labeled_structure)) - 1
input_shape = (labeled_structure.shape[0], labeled_structure.shape[1], len(datacube['time']))

model = create_cnn(input_shape, num_classes)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=1)

def visualize_predictions(y_true, y_pred):
    unique_values = np.unique(y_true)
    unique_values = unique_values[unique_values != 0]
    cmap = plt.cm.get_cmap('tab20', len(unique_values))

    plt.figure(figsize=(20, 10))

    # Display ground truth
    plt.subplot(1, 2, 1)
    plt.imshow(y_true, cmap=cmap, origin='upper')
    plt.title('Ground Truth')

    # Display prediction
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred, cmap=cmap, origin='upper')
    plt.title('Prediction')

    # Create legend
    handles = []
    for plot_id in unique_values:
        label = f'Plot {plot_id}'
        color = cmap(plot_id / len(unique_values))
        handles.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

# Evaluate the model
model.evaluate(x_test, y_test)

# Visualize the results
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)
visualize_predictions(y_test[0], y_pred[0])




