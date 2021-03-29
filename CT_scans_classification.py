# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:42:54 2021

@author: cypri
"""
import CT_processing as CTP
from a.CT_get_model import get_model
import zipfile
import os
import requests
import numpy as np
import random
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
model=get_model()
@tf.function
def rotate(volume):
    def scipy_rotate(volume):
        angles=[-20,-10,-5,5,10,20]
        angle=random.choice(angles)
        volume=ndimage.rotate(volume,angle,reshape=False)
        volume[volume<0]=0
        volume[volume>1]=1
        return volume
    augmented_volume=tf.numpy_function(scipy_rotate,[volume],tf.float32)
    return augmented_volume


def train_preprocess(volume,label):
    volume=rotate(volume)
    #volume=tf.expand_dims(volume,axis=3)
    volume=tf.reshape(volume, shape=(128,128,64 ,1))
    return volume, label

def val_preprocess(volume,label):
   # volume=tf.expand_dims(volume,axis=3)
    volume=tf.reshape(volume, shape=(128,128,64 ,1))
    return volume, label

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

#%%
def cross_validation(abn_scans,norm_scans,abn_labels,norm_labels,model_getter,cross_size):
    all_scores=[]
    data_size=len(abn_scans)+len(norm_scans) 
    k_data_size=data_size//cross_size
    for i in range(cross_size):
        x_train=np.concatenate((norm_scans[:i*k_data_size],norm_scans[(i+1)*k_data_size:] ,abn_scans[:i*k_data_size],abn_scans[(i+1)*k_data_size:]),axis=0)
        y_train=np.concatenate((norm_labels[:i*k_data_size],norm_labels[(i+1)*k_data_size:] ,abn_labels[:i*k_data_size],abn_labels[(i+1)*k_data_size:]),axis=0)
        
        x_val=np.concatenate((norm_scans[i*k_data_size:(i+1)*k_data_size],abn_scans[i*k_data_size:(i+1)*k_data_size]),axis=0)
        y_val=np.concatenate((norm_labels[i*k_data_size:(i+1)*k_data_size],abn_labels[i*k_data_size:(i+1)*k_data_size]),axis=0)
        
        train_loader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
        val_loader=tf.data.Dataset.from_tensor_slices((x_val,y_val))
        batch_size = 2
        
        train_dataset = (
        train_loader.shuffle(len(x_train)).map(train_preprocess).batch(batch_size).prefetch(2))
        
        validation_dataset = (val_loader.shuffle(len(x_val)).map(val_preprocess).batch(batch_size).prefetch(2))
        
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="acc", patience=15)
        model=model_getter()
        epochs = 100
        
        model.fit(train_dataset,epochs=epochs,callbacks=[ early_stopping_cb],)
        val_loss, val_accuracy = model.evaluate(validation_dataset)
        all_scores.append(val_accuracy)
 


    return all_scores

#%%


url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename="CT-0.zip"
filepath = os.path.join(os.getcwd(), filename)
if filename not in [name for name in os.listdir(".")]:
    download_url(url,filepath)


url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename="CT-23.zip"
filepath = os.path.join(os.getcwd(), filename)
if filename not in [name for name in os.listdir(".")]:
    download_url(url,filepath)


#%%
dir_name="CT_data"
if dir_name not in [name for name in os.listdir(".") if os.path.isdir(name)]:
    try:
        os.makedirs(dir_name)
        with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
            z_fp.extractall("./"+dir_name+ "/")

        with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
            z_fp.extractall("./"+dir_name+ "/")

    except Exception as e:
        pass
    
    
normal_scan_paths=[os.path.join(os.getcwd(),"CT_data/CT-0",x) for x in os.listdir("CT_data/CT-0")]

abnormal_scan_paths=[os.path.join(os.getcwd(),"CT_data/CT-23",x) for x in os.listdir("CT_data/CT-23")]

normal_scans=np.array([CTP.process_scan(path) for path in normal_scan_paths])
abnormal_scans=np.array([CTP.process_scan(path) for path in abnormal_scan_paths])

abnormal_labels=np.array([1 for _ in range(len(abnormal_scans))])

normal_labels=np.array([0 for _ in range(len(normal_scans))])


#x_train=np.concatenate((normal_scans[:80], abnormal_scans[:80]),axis=0)
#y_train=np.concatenate((normal_labels[:80], abnormal_labels[:80]),axis=0)
#train_loader=tf.data.Dataset.from_tensor_slices((x_train))
accuracy=cross_validation(abnormal_scans,normal_scans,abnormal_labels,normal_labels,get_model,5)

#%%
model=get_model()
x_train=np.concatenate((normal_scans,abnormal_scans),axis=0)
y_train=np.concatenate((normal_labels,abnormal_labels),axis=0)
        
train_loader=tf.data.Dataset.from_tensor_slices((x_train,y_train))
batch_size = 2
        
train_dataset = (train_loader.shuffle(len(x_train)).map(train_preprocess).batch(batch_size).prefetch(2))

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="acc", patience=15)

epochs = 100
model.fit(train_dataset,epochs=epochs,callbacks=[ early_stopping_cb],)