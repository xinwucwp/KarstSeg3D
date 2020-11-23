import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
#from unetd import mybce
import matplotlib.pyplot as plt
from scipy.interpolate import interpn


def main(argv):
  loadModel(argv[1])
  goFakeValidation()

def loadModel(mk):
  global model
  model = load_model('check/checkpoint.'+mk+'.hdf5')

def goFakeValidation():
  n1,n2,n3=256,256,256
  #seisPath  = "/media/xinwu/disk-2/karstFW/validation/nx/"
  #predPath  = "/media/xinwu/disk-2/karstFW/validation/px/"
  seisPath  = "../data/validation/nx/"
  predPath  = "../data/validation/px/"
  #ks = [100,101,102,103,106,108,110,112,113,114,115,118,119]
  ks = [100,101,102]
  for k in ks:
    fname = str(k)
    gx = loadData(n1,n2,n3,seisPath,fname+'.dat')
    gs = np.reshape(gx,(1,n1,n2,n3,1))
    fp = model.predict(gs,verbose=1)
    fp = fp[0,:,:,:,0]
    ft = np.transpose(fp)
    ft.tofile(predPath+fname+".dat",format="%4")
    os.popen('./goDisplay valid '+fname).read()

def goJie(): 
  fname = "gg.dat"
  n1,n2,n3=320,1024,1024
  fpath = "/media/xinwu/disk-2/karstFW/jie/"
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  gx = np.reshape(gx,(1,n1,n2,n3,1))
  m2 = 512
  m3 = 512
  g1 = gx[:,0:n1,0:n2, 0:m3,:]
  g2 = gx[:,0:n1,0:n2,m3:n3,:]
  f1 = model.predict(g1,verbose=1) #karst prediction
  f2 = model.predict(g2,verbose=1) #karst prediction
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  fx[0:n1,0:n2, 0:m3] = f1[0,:,:,:,0]
  fx[0:n1,0:n2,m3:n3] = f2[0,:,:,:,0]
  fx = np.transpose(fx)
  fx.tofile(fpath+"fp.dat",format="%4")
  os.popen('./goDisplay jie').read()

def goHongliu(): 
  fname = "gg.dat"
  n1,n2,n3=256,256,256
  fpath = "/media/xinwu/disk-2/karstFW/hongliu/"
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  gx = np.reshape(gx,(1,n1,n2,n3,1))
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  fp = model.predict(gx,verbose=1) #fault prediction
  fx[:,:,:] = fp[0,:,:,:,0]
  fx = np.transpose(fx)
  fx.tofile(fpath+"fp.dat",format="%4")
  os.popen('./goDisplay hongliu').read()


def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gm,gs = np.mean(gx),np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def loadDatax(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx


def sigmoid(x):
    s=1.0/(1.0+np.exp(-x))
    return s

def plot2d(gx,fx,fp,at=1,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(133)
  ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
    main(sys.argv)


