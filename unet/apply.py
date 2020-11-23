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
  #goInterp()
  loadModel(argv[1])
  goFakeValidation()
  #goJie()
  #goHongliu()

def loadModel(mk):
  global model
  #model = load_model('checkd3/fseg-'+mk+'.hdf5',custom_objects={'myssim':myssim})
  #model = load_model('check/checkpoint.'+mk+'.hdf5')#,custom_objects={'myce':myce})
  model = load_model('checkr/checkpoint.'+mk+'.hdf5')#,custom_objects={'myce':myce})
  #model = load_model('mse/check2/fseg-'+mk+'.hdf5')#,custom_objects={'myce':myce})
  #model = load_model('checkd4/fseg-'+mk+'.hdf5')#,custom_objects={'myce':myce})

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

def goFakeValidation():
  n1,n2,n3=256,256,256
  #seisPath  = "/media/xinwu/disk-2/karstFW/validation/nx/"
  #predPath  = "/media/xinwu/disk-2/karstFW/validation/px/"
  seisPath  = "../../data/validation/nx/"
  predPath  = "../../data/validation/pr/"
  ks = [100,101,102,103,106,108,110,112,113,114,115,118,119]
  for k in ks:
    fname = str(k)
    gx = loadData(n1,n2,n3,seisPath,fname+'.dat')
    gs = np.reshape(gx,(1,n1,n2,n3,1))
    fp = model.predict(gs,verbose=1)
    fp = fp[0,:,:,:,0]
    ft = np.transpose(fp)
    ft.tofile(predPath+fname+".dat",format="%4")
  #os.popen('./goDisplay valid '+fname).read()

def goBgp(): 
  fname = "gx.dat"
  fpath = "../data/prediction/bgp/"
  #n1,n2,n3=624,701,801
  n1,n2,n3=2501,481,281
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  gx = np.reshape(gx,(1,n1,n2,n3,1))
  m3 = 272
  m2 = 480
  m1 = 832
  gr = gx[:,::3,:,:,:]
  print(np.shape(gr))
  g = gr[:,0:m1,0:m2,0:m3,:]
  f = model.predict(g,verbose=1) #fault prediction
  #f = sigmoid(f)
  l1,l2,l3=np.shape(gr[0,:,:,:,0])
  fx = np.zeros((l1,l2,l3),dtype=np.single)
  fx[0:m1,0:m2,0:m3] = f[0,:,:,:,0]
  fx = np.transpose(fx)
  gt = np.transpose(gr[0,:,:,:,0])
  gt.tofile("../data/prediction/bgp/"+"gxr.dat",format="%4")
  fx.tofile("../data/prediction/bgp/"+"fxr.dat",format="%4")
  os.popen('./goDisplay bgp').read()

def goDyr(): 
  fname = "gr4.dat"
  fpath = "../data/prediction/dy/"
  #n1,n2,n3=624,701,801
  n1,n2,n3=416,344,400
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  gx = np.reshape(gx,(1,n1,n2,n3,1))
  m3 = n3
  m2 = 336
  m1 = n1
  g = gx[:,0:m1,0:m2,0:m3,:]
  f = model.predict(g,verbose=1) #fault prediction
  #f = sigmoid(f)
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  fx[0:m1,0:m2,0:m3] = f[0,:,:,:,0]
  fx = np.transpose(fx)
  fx.tofile("../data/prediction/dy/"+"fpr4.dat",format="%4")
  os.popen('./goDisplay dyr').read()

def goDy(): 
  fname = "gx.dat"
  fpath = "../data/prediction/dy/"
  n1,n2,n3=1251,701,801
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  gx = np.reshape(gx,(1,n1,n2,n3,1))
  m3 = 800
  m2 = 688
  m1 = 400
  l1 = 448
  g1 = gx[:,0:m1,    0:m2,0:m3,:]
  g2 = gx[:,m1:2*m1, 0:m2,0:m3,:]
  #g3 = gx[:,2*m1:2*m1+l1,0:m2,0:m3,:]
  f1 = model.predict(g1,verbose=1) #fault prediction
  f2 = model.predict(g2,verbose=1) #fault prediction
  #f3 = model.predict(g3,verbose=1) #fault prediction
  #f1 = sigmoid(f1)
  #ff2 = sigmoid(f2)
  #ff3 = sigmoid(f3)
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  fx[0:m1,0:m2,0:m3] = f1[0,:,:,:,0]
  fx[m1:2*m1,0:m2,0:m3] = f2[0,:,:,:,0]
  #fx[2*m1:2*m1+l1,0:m2,0:m3] = f3[0,:,:,:,0]
  fx = np.transpose(fx)
  fx.tofile("../data/prediction/dy/"+"fp.dat",format="%4")
  os.popen('./goDisplay dy').read()

def goDyTest(): 
  seismPath = "./data/dy/"
  n3,n2,n1=801,701,1251
  #n3,n2,n1=512,384,128
  gx = np.fromfile(seismPath+'gx.dat',dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gx = gx[100:700,50:650,0:400]
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  '''
  gmin = np.min(gx)
  gmax = np.max(gx)
  gx = gx-gmin
  gx = gx/(gmax-gmin)
  '''
  gx = np.transpose(gx)
  fp = model.predict(np.reshape(gx,(1,400,600,600,1)),verbose=1)
  fp = fp[0,:,:,:,0]
  k1 = 200
  k2 = 300
  k3 = 300
  gx1 = gx[k1,:,:]
  fp1 = fp[k1,:,:]
  gx2 = gx[:,k2,:]
  fp2 = fp[:,k2,:]
  gx3 = gx[:,:,k3]
  fp3 = fp[:,:,k3]
  plot2d(gx1,fp1,fp1,at=1,png='dy/fp1')
  plot2d(gx2,fp2,fp2,at=2,png='dy/fp2')
  plot2d(gx3,fp3,fp3,at=2,png='dy/fp3')
  fp = np.transpose(fp)
  #fp.tofile("data/dy/"+"fp.dat",format="%4")

def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  #gmin,gmax=np.min(gx)/5,np.max(gx)/5
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


