from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pages
import pickle
import random

def getDataY(a, b, (wa,ha), (wb,hb), rez, rezData):

  data1 = a[-wa:]
  data2 = b[:wb]
  size = min(len(data1), len(data2))
  c = 255.0
  for x in range(1,size-1):
    n = (data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c)
    try:
      rezData[n].append((n,rez))
    except:
      rezData[n] = [(n,rez)]

def getDataX(ra, rb, (wa,ha), (wb,hb), rez, rezData):

  data1 = ra[:ha]
  data2 = rb[-hb:]
  size = min(len(data1), len(data2))
  c = 255.0
  for x in range(1,size-1):
    n = (data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c)
    try:
      rezData[n].append((n,rez))
    except:
      rezData[n] = [(n,rez)]

if __name__ == "__main__":
  sx = 10
  sy = 10

  page1 = pages.ImagePage(sx, sy, "SampleDocs/p01.png")
  #page2 = pages.ImagePage(sx, sy, "SampleDocs/text2.jpg")

  dX = {}
  dY = {}
  needX = {}
  needY = {}

  
  for (rx,cx) in page1.states.keys():
    for (ry,cy) in page1.states.keys():
      if cx != cy+1 or rx != ry:
        rez = 1
      else:
        rez = 0
      getDataX(page1.rotDataPieces[(ry,cy)], page1.rotDataPieces[(rx,cx)], page1.states[(ry,cy)].size, page1.states[(rx,cx)].size, rez, dX)

      if rx != ry+1 or cx != cy:
        rez = 1
      else:
        rez = 0
      getDataY(page1.dataPieces[(ry,cy)], page1.dataPieces[(rx,cx)], page1.states[(ry,cy)].size, page1.states[(rx,cx)].size, rez, dY)
  """
  for (rx,cx) in page2.states.keys():
    for (ry,cy) in page2.states.keys():
      if cx != cy+1 or rx != ry:
        rez = 1
      else:
       rez = 0
      getDataX(page2.rotDataPieces[(ry,cy)], page2.rotDataPieces[(rx,cx)], page2.states[(ry,cy)].size, page2.states[(rx,cx)].size, rez, dX)

      if rx != ry+1 or cx != cy:
        rez = 1
      else:
       rez = 0
      getDataY(page2.dataPieces[(ry,cy)], page2.dataPieces[(rx,cx)], page2.states[(ry,cy)].size, page2.states[(rx,cx)].size, rez, dY)
  """
  dataX = []
  dataY = []
  percentX = {}
  percentY = {}
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              if n in dX:
                percentX[n] = (float(sum([x[1] for x in dX[n]])) / len(dX[n]), sum([x[1] for x in dX[n]]), len(dX[n]))
              if n in dY:
                percentY[n] = (float(sum([y[1] for y in dY[n]])) / len(dY[n]), sum([y[1] for y in dY[n]]), len(dY[n]))
              if n in dX:
                dataX += random.sample(dX[n],min(100,len(dX[n])))
              if n in dY:
                dataY += random.sample(dY[n],min(100,len(dY[n])))

  print "percentX"
  print sorted(percentX.items(), key=lambda x: x[1])
  print "percentY"
  print sorted(percentY.items(), key=lambda x: x[1])


  dsX = SupervisedDataSet(6,1)
  dsY = SupervisedDataSet(6,1)

  for d in dataX:
    dsX.addSample(d[0],d[1])

  for d in dataY:
    dsY.addSample(d[0],d[1])

  print "dsX", len(dsX)
  count = {}
  for d in dsX['input']:
    d = tuple(d)
    try:
      count[d] = count[d] + 1
    except:
      count [d] = 1

  print sorted(count.items(), key=lambda x: x[1])
  print len(count)

  print "dsY", len(dsY)
  count = {}
  for d in dsY['input']:
    d = tuple(d)
    try:
      count[d] = count[d] + 1
    except:
      count [d] = 1

  print sorted(count.items(), key=lambda x: x[1])
  print len(count)

  netX = buildNetwork(6,6,1)
  netY = buildNetwork(6,12,1)

  tX = BackpropTrainer(netX, dsX, verbose = True, learningrate=0.00005, batchlearning = True, lrdecay=0.95, momentum = 0.2)
  tX.trainEpochs(250)

  tY = BackpropTrainer(netY, dsY, verbose = True, learningrate=0.00005, batchlearning = True, lrdecay=0.95, momentum = 0.2)
  tY.trainEpochs(250)

  rs = []
  print "BlackGaus"
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              data1 = (i1,i2,i3)
              data2 = (i4,i5,i6)
              rez = 0
              x = 1
              if data1[x] == 0:
                if data2[x-1] == 0:
                  rez -= 1.5
                else:
                  rez += 0.75
                if data2[x+1] == 0:
                  rez -= 1.5
                else:
                  rez += 0.75
                if data2[x] == 0:
                  rez -= 4
                else:
                  rez += 5
              if data2[x] == 0:
                if data1[x-1] == 0:
                  rez -= 1.5
                else:
                  rez += 0.75
                if data1[x+1] == 0:
                  rez -= 1.5
                else:
                  rez += 0.75
                if data1[x] == 0:
                  rez -= 4
                else:
                  rez += 5
              rs.append( (data1, data2, rez))

  print sorted(rs, key=lambda x: x[2])

  rs = []
  print "X"
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              rs.append((n, netX.activate(n)))

  print sorted(rs, key=lambda x: x[1])
  rs = []
  print "Y"
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              rs.append((n, netY.activate(n)))

  print sorted(rs, key=lambda x: x[1])

  with open("nnPickled10",'w') as f:
    pickle.dump((netX,netY),f)
