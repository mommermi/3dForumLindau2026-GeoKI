# data module, michael.mommert@hft-stuttgart.de, 2026, MIT-License

import matplotlib as mpl
import pandas as pd
from torch.utils.data import Dataset
import rasterio as rio
import numpy as np
import torch
import matplotlib.pyplot as plt


# define ESA WorldCover colormap
COLOR_CATEGORIES = [
    (0, 100, 0),
    (255, 187, 34),
    (255, 255, 76),
    (240, 150, 255),
    (250, 0, 0),
    (180, 180, 180),
    (240, 240, 240),
    (0, 100, 200),
    (0, 150, 160),
    (0, 207, 117),
    (250, 230, 160)]
cmap_sen2 = mpl.colors.ListedColormap(np.array(COLOR_CATEGORIES)/255.)

# ESA WorldCover class labels
ewc_label_names = ["tree_cover", "shrubland", "grassland", "cropland", "built-up",
                   "bare/sparse_vegetation", "snow_and_ice","permanent_water_bodies",
                   "herbaceous_wetland", "mangroves","moss_and_lichen"]

# dop colormap
cmap_dop = mpl.colors.ListedColormap(np.array([(0, 0, 0), (150, 100, 0), (200, 200, 0)])/255.)

class Scene():
    """A dataset loading and tiling a single GeoTIFF image."""
    def __init__(self, 
                 data_path):
        """Dataset class constructor

        positional arguments:
        data_path -- string containing the path to the image file

        returns:
        S2Scene object
        """

        # store some definitions
        self.data_path = data_path

        # load Sentinel-2 data
        with rio.open(data_path) as dataset:
            self.data = dataset.read()
            self.crs = dataset.crs
            self.transform = dataset.transform

        # normalize image data
        self.data = self.data.astype(float)
        if 'sen2' in self.data_path:  # Sentinel-2 data
            for i in range(self.data.shape[0]):
                if i < 3:
                    self.data[i] = self.data[i]/3000  # normalize Sentinel-2 data
                else:
                    self.data[i] = self.data[i]/6000  # normalize Sentinel-2 data
        else:  # orthophotos
            self.data /= 255
            self.data -= np.array([0.4046983, 0.43738043, 0.39495268, 0.56818163]).reshape(4, 1, 1)
            self.data /= np.array([0.17502435, 0.16641182, 0.15543586, 0.17817082]).reshape(4, 1, 1)
    
    
    def display(self, pred=None, aoi=None):
        """Display this Sentinel-2 scence; potentially add corresponding prediction.
        
        keyword arguments:
        pred -- prediction tensor, default: None
        aoi -- list of aoi bounds in pixel coordinates: [west, south, east, north]
        """

        if 'sen2' in self.data_path:
            band_order = 'bgri'
            cmap = cmap_sen2
            vmax=11
        else:
            band_order = 'rgbi'
            cmap = cmap_dop
            vmax=2

        if pred is None:
            f, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 10))
            ax = [ax]
        else:
            f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))

        # display Sentinel-2 image
        if aoi:
            img_rgb = self.data[0:3, aoi[3]:aoi[1], aoi[0]:aoi[2]]  # extract RGB for AOI
        else:
            img_rgb = self.data[0:3]  # extract RGB for entire image
        if band_order == 'bgri':
            img_rgb = img_rgb[::-1] # reorder
        img_rgb = np.dstack(img_rgb) # perform deepstack
        img_rgb = np.clip((img_rgb-np.percentile(img_rgb, 1))/(np.percentile(img_rgb, 99)-np.percentile(img_rgb, 1)), 0, 1)
        ax[0].imshow(img_rgb)
        ax[0].set_title('Bilddaten')

        # display prediction, if available
        if pred is not None:
            ax[1].imshow(img_rgb)
            if aoi is not None:
                ax[1].imshow(pred[aoi[3]:aoi[1], aoi[0]:aoi[2]], cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest', alpha=0.8)
            else:
                ax[1].imshow(pred, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest', alpha=0.8)
            ax[1].set_title('Modellausgabe')
