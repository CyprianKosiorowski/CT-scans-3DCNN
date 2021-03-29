# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:13:31 2021

@author: cypri
"""
import numpy as np
import nibabel as nib
from scipy import ndimage
def read_file(filepath):
    scan=nib.load(filepath)
    scan=scan.get_fdata()
    return scan


def normalize(volume,min=-1000,max=400):

    volume[volume<min]=min
    volume[volume>max]=max
    volume=(volume-min)/(max-min)
    volume=volume.astype("float32")
    return volume

def resize_volume(img,desired_size=(128,128,64)):
    current_width=img.shape[0]
    current_height=img.shape[1]
    current_depth=img.shape[-1]
    depth = current_depth / desired_size[2]
    width = current_width / desired_size[0]
    height = current_height / desired_size[1]
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

    
def process_scan(path):
    volume=read_file(path)
    volume=normalize(volume)
    volume=resize_volume(volume)
    return volume
