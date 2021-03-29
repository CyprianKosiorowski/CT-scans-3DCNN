# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:20:13 2021

@author: cypri
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, GlobalAveragePooling3D, Dense, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def get_model(width=128,height=128,depth=64):
    inputs=Input((width,height,depth,1))
    layer=Conv3D(filters=64,kernel_size=3,activation="relu")(inputs)
    layer=MaxPool3D(pool_size=2)(layer)
    layer=BatchNormalization()(layer)
    
    layer=Conv3D(filters=64,kernel_size=3,activation="relu")(layer)
    layer=MaxPool3D(pool_size=2)(layer)
    layer=BatchNormalization()(layer)
    
    layer=Conv3D(filters=128,kernel_size=3,activation="relu")(layer)
    layer=MaxPool3D(pool_size=2)(layer)
    layer=BatchNormalization()(layer)
    
    layer=Conv3D(filters=256,kernel_size=3,activation="relu")(layer)
    layer=MaxPool3D(pool_size=2)(layer)
    layer=BatchNormalization()(layer)
    
    layer = GlobalAveragePooling3D()(layer)
    layer = Dense(units=512, activation="relu")(layer)
    layer = Dropout(0.3)(layer)

    outputs = Dense(units=1, activation="sigmoid")(layer)

    # Define the model.
    model = Model(inputs, outputs, name="3Dcnn")
    initial_learning_rate = 0.0001
    lr_schedule = ExponentialDecay(initial_learning_rate,
                                                                  decay_steps=100000, decay_rate=0.96, staircase=True)
        
    model.compile(loss="binary_crossentropy",optimizer=Adam(learning_rate=lr_schedule),metrics=["acc"],)

    return model    
