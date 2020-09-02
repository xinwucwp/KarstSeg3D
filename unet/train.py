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
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log5', **kwargs):
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

def rotateVectors(theta,n1,n2,n3,x1,x2):
  cost = np.cos(theta)
  sint = np.sin(theta)
  x1r = cost*x1-sint*x2
  x2r = sint*x1+cost*x2
  return x1r,x2r
# get training patches
def getTrainingCubes(n1,n2,n3,nd,sxpath,kxpath):
  stride = 50
  p1,p2,p3=64,64,64
  m1 = int((n1-p1-10)/stride)
  m2 = int((n2-p2-10)/stride)
  m3 = int((n3-p3-10)/stride)
  files = os.listdir(sxpath)
  #nd = len(files)
  #mp = m1*m2*m3*nd
  mp = m1*m2*m3*nd*4
  x = np.zeros((mp, p1, p2, p3, 1),dtype=np.single)
  y = np.zeros((mp, p1, p2, p3, 1),dtype=np.single)
  i = 0
  for fk in range(0,nd,1):
    fn = str(fk)+'.dat'
    print(fn)
  #for fn in files:
    sx = np.fromfile(sxpath+fn,dtype=np.single)
    kx = np.fromfile(kxpath+fn,dtype=np.single)

    sx = np.reshape(sx,(n1,n2,n3))
    kx = np.reshape(kx,(n1,n2,n3))

    sx = np.transpose(sx)
    kx = np.transpose(kx)

    xm = np.mean(sx)
    xs = np.std(sx)
    sx = sx-xm
    sx = sx/xs
    for i3 in range(m3):
      for i2 in range(m2):
        for i1 in range(m1):
          k1 = i1*stride+5
          k2 = i2*stride+5
          k3 = i3*stride+5
          sxi = sx[k1:k1+p1,k2:k2+p2,k3:k3+p3]
          kxi = kx[k1:k1+p1,k2:k2+p2,k3:k3+p3]

          '''
          x[i,]  = np.reshape(nxi,(p1,p2,p3,1))
          y1[i,] = np.reshape(fxi,(p1,p2,p3,1))
          y2[i,] = np.reshape(cxi,(p1,p2,p3,1))
          y3[i,:,:,:,0] = u1i[:,:,:]
          y3[i,:,:,:,1] = u2i[:,:,:]
          y3[i,:,:,:,2] = u3i[:,:,:]
          i = i+1
          '''
          for k in range(4):
            x[i,] = np.reshape(np.rot90(sxi,k,(2,1)),(p1,p2,p3,1))
            y[i,] = np.reshape(np.rot90(kxi,k,(2,1)),(p1,p2,p3,1))
            i = i+1
  return x,y

def shuffle_in_unison(a, b, c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(c)
#############################################################

# checkpoint
filepath="check/checkpoint.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
        verbose=1, save_best_only=False, mode='max')
logging = TrainValTensorBoard()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-8)
callbacks_list = [checkpoint, logging, reduce_lr]

sxpath = '/media/xinwu/disk-2/karst/seis/'
kxpath = '/media/xinwu/disk-2/karst/karst/'
n1,n2,n3,nd=512,512,512,5
x,y=getTrainingCubes(n1,n2,n3,nd,sxpath,kxpath)
print("data prepared, ready to train!")

from unet3 import *
model = unet(input_size=(None, None, None,1))
#model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])
# Fit the model
history = model.fit(x, y, validation_split=0.1, epochs=25, batch_size=8, callbacks=callbacks_list, verbose=1)
model.save('karst.hdf5')
print(history)


