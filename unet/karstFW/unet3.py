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


def unet(pretrained_weights = None,input_size = (None,None,None,None)):
    def _residual_block(inputs, feature_dim=32):
      x = Conv3D(feature_dim, (3, 3, 3), padding="same", kernel_initializer="he_normal")(inputs)
      x = BatchNormalization()(x)
      x = PReLU(shared_axes=[1, 2, 3])(x)
      x = Conv3D(feature_dim, (3, 3, 3), padding="same", kernel_initializer="he_normal")(x)
      x = BatchNormalization()(x)
      m = Add()([x, inputs])
      m = PReLU(shared_axes=[1, 2, 3])(m)
      return m

    input_img = Input(shape=input_size,name='input_image')
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(input_img)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Conv3D(512, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(512, (3,3,3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv3], axis=4)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv2], axis=4)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv1], axis=4)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)

    '''
    conv8  = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)
    conv9  = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)
    conv10 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)
    '''
    conv8 = _residual_block(conv7,feature_dim=16)
    conv8 = _residual_block(conv8,feature_dim=16)
    print(conv8._keras_shape)

    o1 = Conv3D(1, (1,1,1), activation='sigmoid',name='o1')(conv8)

    '''
    model = Model(inputs=input_img, outputs=[o1,o2])
    model.compile(optimizer = Adam(lr = 1e-4),
      loss = {'o1':cross_entropy_balanced,'o2':'mean_squared_error'}, 
      metrics = {'o1':'accuracy','o2':'mse'})
    '''
    model = Model(inputs=input_img, outputs=o1)
    model.compile(optimizer = Adam(lr = 1e-4),loss ='binary_crossentropy', metrics = ['accuracy'])
    '''
    model.compile(optimizer = Adam(lr = 1e-4),
      loss = {'o1':cross_entropy_balanced,'o2':'mean_squared_error','o3':'mean_squared_error'}, 
      metrics = {'o1':'accuracy','o2':'mse','o3':'mse'})
    '''
    return model

def cosine_similarity(y_true, y_pred):
    a11 = tf.reduce_sum(tf.square(y_true),-1)
    a22 = tf.reduce_sum(tf.square(y_pred),-1)
    a12 = tf.reduce_sum(tf.multiply(y_true,y_pred),-1)
    cos = a12/tf.sqrt(tf.multiply(a11,a22))
    return tf.reduce_mean(10.-10*cos)

def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, 
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
