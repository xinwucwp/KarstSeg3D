#############################################################################
"""
Demo of demostrating fault results
Author: Xinming Wu, USTC
Version: 2020.03.26
"""
from datUtils import * 
#############################################################################

plotOnly = True
plotOnly = False

def main(args):
  #goPrecisionRecall()
  #goCoherence()
  goJiePrecisionRecall()
def goValid(fname):
  global pngDir
  setupForSubset("validation")
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage("nx/"+fname)
  fp = readImage("px/"+fname)
  ks = [220,212,50]
  vt=[-0.0,-0.0,0.0]
  ae=[125,20]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
          clab="Karst probability",png="fp")


def goPrecisionRecall():
  global pngDir
  setupForSubset("validation")
  pngDir = getPngDir()
  ks = [100,102,103,108,110,112,118,119]
  kxs,chs,cvs,kps = [],[],[],[]
  for k in ks:
    fname = str(k)
    kx = readImage("kx/"+fname)
    ch = readImage("chx/"+fname)
    cv = readImage("cvx/"+fname)
    kp = readImage("pr1/"+fname)
    ch = sub(1,pow(ch,6))
    cv = clip(min(cv),0.02,cv)
    cv = sub(cv,min(cv))
    cv = div(cv,max(cv))
    cv = sub(1,pow(cv,6))
    #cv = sqrt(cv)
    chs.append(ch)
    cvs.append(cv)
    kps.append(kp)
    kxs.append(kx)
  pp = Precision()
  pxs = [cvs,chs,kps]
  fpr = []
  tpr = []
  pcr = []
  for px in pxs:
    pr = pp.getPrecisionRecall(0.0,0.05,kxs,px)
    pcr.append(pr[0])
    tpr.append(pr[1])
    fpr.append(pr[2])
  plot1s(tpr,pcr,hint=0.2, hw=500,vw=400,
          hlabel="Recall",vlabel="Precision",png="pr-curve")
  plot1s(fpr,tpr,hint=0.2,hw=500,vw=400,
        hlabel="False positive rate",vlabel="True positive rate",png="roc-curve")

  '''
  pr = pp.getPrecisionRecall(0.0,0.05,kx,kp)
  print(pr[0])
  print(pr[1])
  plot1(pr[1],pr[0],hlabel="Recall",vlabel="Precision",png="pr-curve")
  plot1(pr[2],pr[1],vmin=0.6,hint=0.2,hw=500,vw=500,
        hlabel="False positive rate",vlabel="True positive rate",png="roc-curve")
  '''

def goJiePrecisionRecall():
  global pngDir
  setupForSubset("jie")
  pngDir = getPngDir()
  gx = readImage("gg")
  lx = readImage("lx")
  kx = readImage("kc")
  ch = readImage("cs")
  k1 = readImage("k1c")
  n3 = len(gx)
  n2 = len(gx[0])
  n1 = len(gx[0][0])
  lx[752] = lx[753]
  k2 = 346
  k3 = 752
  rgf = RecursiveGaussianFilterP(2)
  #rgf.applyX0X(kx,kx)
  #rgf.applyXX0(kx,kx)
  #rgf.apply000(ch,ch)
  gxs = zerofloat(n1,n2+n3)
  lxs = zerofloat(n1,n2+n3)
  cvs = zerofloat(n1,n2+n3)
  chs = zerofloat(n1,n2+n3)
  kps = zerofloat(n1,n2+n3)
  k = 0
  for i3 in range(n3):
    cvs[k]=k1[i3][k2]
    chs[k]=ch[i3][k2]
    kps[k]=kx[i3][k2]
    lxs[k]=lx[i3][k2]
    gxs[k]=gx[i3][k2]
    k = k+1
  for i2 in range(n2):
    cvs[k]=k1[k3][i2]
    chs[k]=ch[k3][i2]
    kps[k]=kx[k3][i2]
    lxs[k]=lx[k3][i2]
    gxs[k]=gx[i3][k2]
    k = k+1
  print max(lxs)
  lxs = clip(0,1,lxs)
  chs = clip(0,1,chs)
  chs = pow(chs,4)
  chs = sub(1,chs)
  cvs = clip(min(cvs),0.,cvs)
  cvs = sub(cvs,min(cvs))
  cvs = div(cvs,max(cvs))
  cvs = sub(1,pow(cvs,6))

  cleanKarst(gxs,chs)
  pp = Precision()
  pxs = [cvs,chs,kps]
  fpr = []
  tpr = []
  pcr = []
  for px in pxs:
    pr = pp.getPrecisionRecall(0.0,0.05,lxs,px)
    pcr.append(pr[0])
    tpr.append(pr[1])
    fpr.append(pr[2])
  plot1s(tpr,pcr,hint=0.2, hw=500,vw=400,
          hlabel="Recall",vlabel="Precision",png="jie-pr-curve")
  plot1s(fpr,tpr,hint=0.2,hw=500,vw=400,
        hlabel="False positive rate",vlabel="True positive rate",png="jie-roc-curve")

  '''
  pr = pp.getPrecisionRecall(0.0,0.05,kx,kp)
  print(pr[0])
  print(pr[1])
  plot1(pr[1],pr[0],hlabel="Recall",vlabel="Precision",png="pr-curve")
  plot1(pr[2],pr[1],vmin=0.6,hint=0.2,hw=500,vw=500,
        hlabel="False positive rate",vlabel="True positive rate",png="roc-curve")
  '''

def cleanKarst(gx,kx):
  m2 = len(gx)
  m1 = len(gx[0])
  zeros = zerofloat(m1)
  for i2 in range(m2):
    sk = sum(gx[i2])
    if(sk==0):
      for k2 in range(-5,5,1):
        j2 = min(max(i2+k2,0),m2-1)
        kx[j2] = zeros

def goCoherence():
  setupForSubset("validation")
  pngDir = getPngDir()
  sig1,sig2 = 6,4 
  ks = [100,101,102,103,106,108,110,112,113,114,115,118,119]
  for k in ks:
    print k
    sxfile = str(k)
    sx = readImage("nx/"+sxfile)
    n3 = len(sx)
    n2 = len(sx[0])
    n1 = len(sx[0][0])
    lof = LocalOrientFilter(8,2)
    ets = lof.applyForTensors(sx)
    ets.setEigenvalues(0.01,1,1)
    lsf = LocalSmoothingFilter()
    ref1 = RecursiveExponentialFilter(sig1)
    ref1.setEdges(RecursiveExponentialFilter.Edges.OUTPUT_ZERO_SLOPE)
    gn = zerofloat(n1,n2,n3)
    gd = zerofloat(n1,n2,n3)
    # compute the numerator of coherence
    lsf.apply(ets,sig2,sx,gn)
    gn = mul(gn,gn)
    ref1.apply1(gn,gn)
    # compute the denominator of coherence
    lsf.apply(ets,sig2,mul(sx,sx),gd)
    ref1.apply1(gd,gd)
    cs = div(gn,gd)
    writeImage("chx/"+str(k),cs)

#############################################################################
def plot1(f,g,vmin=0.0,hint=0.1,hw=400,vw=500,
        hlabel="Precision",vlabel="Recall",png=None):
  sp = SimplePlot()
  pv = sp.addPoints(f,g)
  pv.setLineWidth(3)
  sp.setVLimits(vmin,1.0)
  sp.setHLimits(min(f),1.0)
  sp.setHInterval(hint)
  sp.setVLabel(vlabel)
  sp.setHLabel(hlabel)
  sp.setSize(hw,vw)
  if pngDir and png:
    sp.paintToPng(720,3.33,pngDir+png+".png")
def plot1s(f,g,vmin=0.0,hint=0.1,hw=400,vw=500,
        hlabel="Precision",vlabel="Recall",png=None):
  sp = SimplePlot()
  nc = len(f)
  colors=[Color.BLACK,Color.BLUE,Color.RED,Color.GREEN,
          Color.YELLOW,Color.ORANGE,Color.MAGENTA,Color.RED]
  for ic in range(nc):
    pv = sp.addPoints(f[ic],g[ic])
    pv.setLineWidth(3)
    pv.setLineColor(colors[ic])
  sp.setVLimits(vmin,1.05)
  sp.setHLimits(min(f),1.05)
  sp.setHInterval(hint)
  sp.setVLabel(vlabel)
  sp.setHLabel(hlabel)
  sp.setSize(hw,vw)
  sp.setBackground(Color.LIGHT_GRAY)
  if pngDir and png:
    sp.paintToPng(720,3.33,pngDir+png+".png")

# Run the function main on the Swing thread
import sys
class _RunMain(Runnable):
  def __init__(self,main):
    self.main = main
  def run(self):
    self.main(sys.argv)
def run(main):
  SwingUtilities.invokeLater(_RunMain(main)) 
run(main)
