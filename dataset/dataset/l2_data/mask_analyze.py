import numpy as np
import cv2
from osgeo import gdal
import pdb

def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    return img
lem2_mask = cv2.imread('TrainTestMask.tif', 0).astype(np.uint8)
print(lem2_mask.shape)

lem2_label = load_image('labels/20200912_S1.tif').astype(np.uint8)

lem2_label_masked = lem2_label.copy()
lem2_label_masked[lem2_mask!=2] = 0

print("np.unique(lem2_label, return_counts=True)",np.unique(lem2_label, return_counts=True))

print("np.unique(lem2_label_masked, return_counts=True)",np.unique(lem2_label_masked, return_counts=True))
