# Simplified unet for fault segmentation
# The original u-net architecture is more complicated than necessary 
# for our task of fault segmentation.
# We significanlty reduce the number of layers and features at each 
# layer to save GPU memory and computation but still preserve high 
# performace in fault segmentation.

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights = None,input_size = (None,None,None,1)):
    nf1 = 32
    nf2 = nf1*2
    nf3 = nf2*2
    nf4 = nf3*2
    nf5 = nf4*2
    input_img = Input(shape=input_size,name='input_image')
    conv1 = Conv3D(nf1, (3,3,3), activation='relu', padding='same')(input_img)
    conv1 = Conv3D(nf1, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(nf2, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(nf2, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(nf3, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(nf3, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Conv3D(nf4, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(nf4, (3,3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,2))(conv4)

    conv5 = Conv3D(nf5, (3,3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(nf5, (3,3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv4], axis=-1)
    conv6 = Conv3D(nf4, (3,3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(nf4, (3,3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv3], axis=-1)
    conv7 = Conv3D(nf3, (3,3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(nf3, (3,3,3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2,2,2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(nf2, (3,3,3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(nf2, (3,3,3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2,2,2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(nf1, (3,3,3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(nf1, (3,3,3), activation='relu', padding='same')(conv9)

    o1 = Conv3D(1, (1,1,1), activation='sigmoid',name='o1')(conv9)

    model = Model(inputs=input_img, outputs=o1)
    model.compile(optimizer = Adam(lr = 1e-4),loss ='binary_crossentropy', metrics = ['accuracy'])

    return model

