#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import argparse, sys, os, errno
import logging
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import h5py
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='img_array')
parser.add_argument('-t', dest='patient_id')
args = parser.parse_args()

def plot_3d(image,path,threshold = -300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    verts, faces, _, _ = measure.marching_cubes(p, level = threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.savefig(path)

f = h5py.File(args.img_array)
a = np.loadtxt(args.patient_id,dtype = 'S')
for i in range(a.shape[0]):
    index = a[i]
    img = f['preprocess'+index][:,:,:]
    plot_3d(img,index,-200,)



