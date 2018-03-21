#! /usr/bin/env python
import argparse, sys, os, errno
import numpy as np
import os
import csv
from glob import glob
import pandas as pd
import SimpleITK as sitk
from skimage import measure,morphology
from sklearn.cluster import KMeans
from skimage.transform import resize
import seaborn as sns
import h5py
from skimage.segmentation import clear_border
from skimage import morphology
from skimage import measure
from skimage import filters
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='img_array')
parser.add_argument('-t', dest='patient_id')
parser.add_argument('-o', dest='output_file')
args = parser.parse_args()

#we have had the vcenter before
vcenter = -np.loadtxt('preprocess/vcenter.txt')


luna_subset_path = 'data/train_set/'
file_list=glob(luna_subset_path+"*.mhd")

def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

# The locations of the nodes
df_node = pd.read_csv("data/annotations.csv")
df_node["file"] = df_node["seriesuid"].apply(get_filename)
df_node = df_node.dropna()

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
        Center : centers of circles px -- list of coordinates x,y,z
        diam : diameters of circles px -- diameter
        widthXheight : pixel dim of image
        spacing = mm/px conversion rate np array x,y,z
        origin = x,y,z mm np.array
        z = z position of slice in world coordinates mm
        '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    '''
        matrix must be a numpy array NXN
        Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

# keep slices for two kinds: according to the diameter
# keep slices for two kinds: according to the diameter
for i in tqdm(range(975)):
    # load the data once
    itk_img = sitk.ReadImage(df_node['file'][i])
    img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    # go through all nodes (why just the biggest?)
    node_x = df_node["coordX"][i]
    node_y = df_node["coordY"][i]
    node_z = df_node["coordZ"][i]
    diam = df_node["diameter_mm"][i]
    dd = int(np.ceil(diam))
    imgs = np.ndarray([dd,height,width],dtype=np.float32)
    masks = np.ndarray([dd,height,width],dtype=np.uint8)
    center = np.array([node_x, node_y, node_z])   # nodule center
    v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
    for j, i_z in enumerate(np.arange(int(v_center[2]-diam/2.0),
                                      int(v_center[2]+diam/2.0)).clip(0, num_z-1)): # clip prevents going out of bounds in Z
        mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
                         width, height, spacing, origin)
        masks[j] = mask
        imgs[j] = img_array[i_z]
    imshape = imgs.shape[0]
    mashape = masks.shape[0]
    np.save(os.path.join("preprocess/forunet/images/diameter/images_%s_%d.npy" % (df_node['node'][i],imshape)),imgs)
    np.save(os.path.join("preprocess/forunet/masks/diameter/masks_%s_%d.npy"  % (df_node['node'][i],mashape)),masks)






