from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)
#import cv2
import os
import random
import numpy as np
import skimage
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from utils import DataGenerator
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)
    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        logs.update({'lr': keras.eval(self.model.optimizer.lr)})
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

#############################################################

# checkpoint
filepath="check/checkpoint.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
        verbose=1, save_best_only=False, mode='max')
logging = TrainValTensorBoard()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-8)
callbacks_list = [checkpoint, logging, reduce_lr]

sxpath = '/media/xinwu/disk-2/karstFW/noise/'
kxpath = '/media/xinwu/disk-2/karstFW/karst/'
n1,n2,n3,nd=256,256,256,5
tdata_ids = range(0,100)
vdata_ids = range(100,120)
params = {'batch_size':1, 'dim':(n1,n2,n3),'n_channels':1,'shuffle':True}
train = DataGenerator(dpath=sxpath,fpath=kxpath,data_IDs=tdata_ids,**params)
valid = DataGenerator(dpath=sxpath,fpath=kxpath,data_IDs=vdata_ids,**params)

from unet3 import *
model = unet(input_size=(None, None, None,1))
# Fit the model
history = model.fit_generator(generator=train,validation_data=valid,epochs=25, callbacks=callbacks_list, verbose=1)
print(history)


