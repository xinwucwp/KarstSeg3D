"""
Jython utilities for cig.
Author: Xinming Wu, USTC
Version: 2020.03.28
"""
from common import *

#############################################################################
# Internal constants

#_datdir = "/media/xinwu/disk-2/karstFW/"
_datdir = "../data/"
_pngdir = "../png/"

#############################################################################
# Setup

def setupForSubset(name):
  """
  Setup for a specified directory includes:
    seismic directory
    samplings s1,s2
  Example: setupForSubset("pnz")
  """
  global pngDir
  global seismicDir
  global s1,s2,s3
  global n1,n2,n3
  if name=="jie":
    """ jie """
    print "setupForSubset: jie"
    pngDir = _pngdir+"jie/"
    seismicDir = _datdir+"jie/"
    n1,n2,n3 = 320,1024,1024
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="validation":
    print "setupForSubset: validation"
    pngDir = _pngdir+"validation/"
    seismicDir = _datdir+"validation/"
    n1,n2,n3 = 256,256,256
    d1,d2,d3 = 1.0,1.0,1.0 
    f1,f2,f3 = 0.0,0.0,0.0 # = 0.000,0.000,0.000
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="hongliu":
    print "setupForSubset: hongliu"
    pngDir = _pngdir+"hongliu/"
    seismicDir = _datdir+"hongliu/"
    n1,n2,n3 = 256,256,256
    d1,d2,d3 = 1.0,1.0,1.0 
    f1,f2,f3 = 0.0,0.0,0.0 # = 0.000,0.000,0.000
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  else:
    print "unrecognized subset:",name
    System.exit

def getSamplings():
  return s1,s2,s3

def getSeismicDir():
  return seismicDir

def getPngDir():
  return pngDir

#############################################################################
# read/write images
def readImageChannels(basename):
  """ 
  Reads three channels of a color image
  """
  fileName = seismicDir+basename+".jpg"
  il = ImageLoader()
  image = il.readThreeChannels(fileName)
  return image
def readColorImage(basename):
  """ 
  Reads three channels of a color image
  """
  fileName = seismicDir+basename+".jpg"
  il = ImageLoader()
  image = il.readColorImage(fileName)
  return image

def readImage2D(n1,n2,basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def readImage(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImage3DB(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.BIG_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImageL(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readInts(image)
  ais.close()
  return image

def readImage2DL(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image


def readImage1D(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def readImage1L(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def writeImage(basename,image):
  """ 
  Writes an image to a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  #aos.writeBytes(image)
  aos.close()
  return image

def writeImagex(fname,image):
  """ 
  Writes an image to a file with specified basename
  """
  fileName = fname+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  #aos.writeBytes(image)
  aos.close()
  return image


def writeImageL(basename,image):
  """ 
  Writes an image to a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  print fileName
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

#############################################################################
# read/write fault skins

def skinName(basename,index):
  return basename+("%05i"%(index))
def skinIndex(basename,fileName):
  assert fileName.startswith(basename)
  i = len(basename)
  return int(fileName[i:i+5])

def listAllSkinFiles(basename):
  """ Lists all skins with specified basename, sorted by index. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  return fileNames

def removeAllSkinFiles(basename):
  """ Removes all skins with specified basename. """
  fileNames = listAllSkinFiles(basename)
  for fileName in fileNames:
    File(seismicDir+fileName).delete()

def readSkin(basename,index):
  """ Reads one skin with specified basename and index. """
  return FaultSkin.readFromFile(seismicDir+skinName(basename,index)+".dat")

def getSkinFileNames(basename):
  """ Reads all skins with specified basename. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  return fileNames

def readSkins(basename):
  """ Reads all skins with specified basename. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  skins = []
  for iskin,fileName in enumerate(fileNames):
    index = skinIndex(basename,fileName)
    skin = readSkin(basename,index)
    skins.append(skin)
  return skins

def writeSkin(basename,index,skin):
  """ Writes one skin with specified basename and index. """
  FaultSkin.writeToFile(seismicDir+skinName(basename,index)+".dat",skin)

def writeSkins(basename,skins):
  """ Writes all skins with specified basename. """
  for index,skin in enumerate(skins):
    writeSkin(basename,index,skin)

from org.python.util import PythonObjectInputStream
def readObject(name):
  fis = FileInputStream(seismicDir+name+".dat")
  ois = PythonObjectInputStream(fis)
  obj = ois.readObject()
  ois.close()
  return obj
def writeObject(name,obj):
  fos = FileOutputStream(seismicDir+name+".dat")
  oos = ObjectOutputStream(fos)
  oos.writeObject(obj)
  oos.close()
