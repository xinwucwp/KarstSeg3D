import numpy as np
import keras
import random
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(512,512,512), 
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    #self.indexes = np.arange(len(self.data_IDs))
    self.indexes = []
    for k in range(len(self.data_IDs)):
      self.indexes.append(k)
      self.indexes.append(k)
      self.indexes.append(k)
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    ms = []
    ms.append(64)
    ms.append(128)
    ms.append(160)
    #ms.append(256)
    m1 = 128#ms[random.randint(0,1)]
    m2 = 128#ms[random.randint(0,1)]
    m3 = 128
    n1,n2,n3=self.dim
    X = np.zeros((4, m1, m2, m3, self.n_channels),dtype=np.single)
    Y = np.zeros((4, m1, m2, m3, self.n_channels),dtype=np.single)

    gx  = np.fromfile(self.dpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    kx  = np.fromfile(self.fpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    kx = np.reshape(kx,self.dim)
    kx = np.transpose(kx)

    gx = np.reshape(gx,self.dim)
    gx = np.transpose(gx)

    k1 = random.randint(0,n1-m1-1)
    k2 = random.randint(0,n2-m2-1)
    k3 = random.randint(0,n3-m3-1)
    gx = gx[k1:k1+m1,k2:k2+m2,k3:k3+m3]
    kx = kx[k1:k1+m1,k2:k2+m2,k3:k3+m3]

    mx = np.mean(gx)
    xs = np.std(gx)
    gx = gx-mx
    gx = gx/xs
    # Generate data
    for k in range(4):
      X[k,] = np.reshape(np.rot90(gx,k,(2,1)), (m1, m2, m3, self.n_channels))
      Y[k,] = np.reshape(np.rot90(kx,k,(2,1)), (m1, m2, m3, self.n_channels))  
    return X,Y
