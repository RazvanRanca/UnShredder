import random
import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import math
from itertools import groupby

delta = 0.1
inf = -1000 # float("inf")
nnD = 0.87

if False:
  with open("nnPickled10",'r') as f:
    (netX,netY) = pickle.load(f)

  dnx = {}
  dny = {}

  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              dnx[n] = netX.activate(n)
              dny[n] = netY.activate(n)


def picker(s, page, px = None, py = None, prior = None):
  if s == "bit":
    return calcBitCost(page)
  elif s == "gaus":
    return calcGausCost(page)
  elif s == "blackBit":
    return calcBlackBitCost(page)
  elif s == "blackGaus":
    return calcBlackGausCost(page)
  elif s == "blackBigGaus":
    return calcBlackBigGausCost(page)
  elif s == "rand":
    return calcRandCost(page)
  elif s == "blackRow":
    return calcBlackRowCost(page)
  elif s == "blackRowGaus":
    return calcBlackRowGausCost(page)
  elif s == "nn":
    return calcNNCost(page)
  elif s == "percent":
    return calcPercentCost(page, px, py)
  elif s == "prediction":
    return calcPredictionCost(page, px, py, prior)

def indivPicker(s, a,b, tp, page, px = None, py = None, prior = None, selective = False): #selective is a flag showing wether to give infinity of zero for stuff like blank on blank
  blank = page.blank
  pieces = page.states
  if s == "bit":
    if tp == "x":
      return imgBitCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
    elif tp == "y":
      return imgBitCostY(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
  elif s == "gaus":
    if tp == "x":
      return imgGausCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
    elif tp == "y":
      return imgGausCostY(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
  elif s == "blackBit":
    if tp == "x":
      return imgBlackBitCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
    elif tp == "y":
      return imgBlackBitCostY(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
  elif s == "blackGaus":
    if tp == "x":
      return imgBlackGausCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
    elif tp == "y":
      return imgBlackGausCostY(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective=selective)
  elif s == "rand":
    mult = max(sx,sy) * 100
    if tp == "x":
      return random.random() * 1.0/page.sizeY * mult
    elif tp == "y":
      return random.random() * 1.0/page.sizeX * mult
  elif s == "blackRow":
    if tp == "x":
      return imgBlackBitCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective)
    elif tp == "y":
      return imgBitCostY(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective)
  elif s == "blackRowGaus":
    if tp == "x":
      return imgBlackGausCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective)
    elif tp == "y":
      return imgGausCostY(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, selective)
  elif s == "cached":
    if tp == "x":
      return page.costX[a,b]
    elif tp == "y":
      return page.costY[a,b]
  elif s == "prediction":
    if tp == "x":
      return imgPredCostX(page.rotDataPieces[a], page.rotDataPieces[b], pieces[a].size, pieces[b].size, page.prx, page.prior)
    elif tp == "y":
      return imgPredCostY(page.dataPieces[a], page.dataPieces[b], pieces[a].size, pieces[b].size, page.pry, page.prior)

def evaluateCost(page, sx, sy): # calculate percent of edges whose best match given by the cost function is the true neighbour and ammount of error
  costX = page.costX
  costY = page.costY  
  bestX = {}
  bestY = {}
  #print costX
  #print costY
  for (k1,k2),v in costX.items():
    if k1 not in bestX or v < bestX[k1][1]:
      bestX[k1] = (k2, v, 1)
    elif v == bestX[k1][1]:
      bestX[k1] = (bestX[k1][0],bestX[k1][1],bestX[k1][2]+1)

  for (k1,k2),v in costY.items():
    if k1 not in bestY or v < bestY[k1][1]:
      bestY[k1] = (k2, v, 1)
    elif v == bestY[k1][1]:
      bestY[k1] = (bestY[k1][0], bestY[k1][1], bestY[k1][2]+1)

  correct = 0
  count = 0
  error = 0
  for x in range(sx):
    for y in range(sy):
      cx = x + 1
      cy = y
      if cx < sx:
        count += 1
        if costX[((y,x),(cy,cx))] == bestX[(y,x)][1]:
          correct += 1.0/bestX[(y,x)][2]
          #print "X", y, x, bestX[(y,x)][2]
        else:
          #print "XX", y,x
          error += abs(costX[((y,x),(cy,cx))] - bestX[(y,x)][1])

      cx = x
      cy = y + 1
      if cy < sy:
        count += 1
        if costY[((y,x),(cy,cx))] == bestY[(y,x)][1]:
          correct += 1.0/bestY[(y,x)][2]
          #print "Y", y, x, bestY[(y,x)][2]
        else:
          #print "YY", y,x
          error += abs(costY[((y,x),(cy,cx))] - bestY[(y,x)][1])
  #print correct, count
  #print sorted(bestX.items())
  #print sorted(bestY.items())
  return float(correct) / count, float(error) / count

def normalizeCost(cost):
  count = {}
  for v in cost.values():
    try:
      count[v] = dist[v]+1
    except:
      count[v] = 1

  nCost = {}
  for (k,v) in cost.items():
    nCost[k] = float(v) / count[v]

  return nCost

def processCostX(page):
  dist = {}
  for v in page.costX.values():
    try:
      dist[v] = dist[v]+1
    except:
      dist[v] = 1

  sProb = {}
  states = page.getAllStates()
  noStates = len(states)

  for piece in states:
    pDist = {}  # get P(piece dist | global dist)

    for v in page.pieceDistX(piece).values():
      try:
        pDist[v] = pDist[v]+1
      except:
        pDist[v] = 1
    distProb = 1.0

    for v in pDist:
      distProb *= pDist[v]*dist[v]
    
    scoreProb = 1.0   # get P(score | piece dist)
    scores = page.pieceDistX(piece)
    for s in  scores:
      sProb[s]= distProb * pDist[scores[s]] * (scores[s] + 1)

  return sProb

def processCostY(page):
  dist = {}
  for v in page.costY.values():
    try:
      dist[v] = dist[v]+1
    except:
      dist[v] = 1

  sProb = {}
  states = page.getAllStates()
  noStates = len(states)

  for piece in states:
    pDist = {}  # get P(piece dist | global dist)

    for v in page.pieceDistY(piece).values():
      try:
        pDist[v] = pDist[v]+1
      except:
        pDist[v] = 1
    distProb = 1.0

    for v in pDist:
      distProb *= pDist[v]*dist[v]
    
    scoreProb = 1.0   # get P(score | piece dist)
    scores = page.pieceDistY(piece)
    for s in  scores:
      sProb[s]= distProb * pDist[scores[s]] * (scores[s] + 1)

   
  #norm = 1000000.0/sum(sProb.values()) # just normalizing the numbers to reasonable values
  #sProb = dict(map(lambda (k,v):(k,v*norm), sProb.items()))
  #print sorted(sProb.items(), key=lambda x: x[1])[:1000]
  
  return sProb

def calcPredictionCost(page, px, py, prior):
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgPredCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size, px, prior)
      costY[(y, x)] = imgPredCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size, py, prior)

  #print costX[((1,0),(0,0))]
  #print costX[((1,0),(1,1))]
  #a = costX[((0,1),(0,2))]
  #b = costX[((0,1),(-42,-42))]
  #print math.fsum(a[0]), math.fsum(a[1]), math.fsum(b[0]), math.fsum(b[1])
  #print  math.fsum(map(math.log,a)), [(key,len(list(group))) for key, group in groupby(sorted(a))]
  #print  math.fsum(map(math.log,b)), [(key,len(list(group))) for key, group in groupby(sorted(b))]
  #assert False
  return costX, costY

def imgPredCostY(a, b, (wa,ha),(wb,hb), (pl,pr), prior, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = a[-wa:]
  #data0 = a[-2*wa:-wa]
  data2 = b[:wb]
  #if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
  #  return inf
  size = min(len(data1), len(data2))
  c = 255.0
  first = 1 - prior
  if data2[0] == 0:
    first = prior
  rezl = [math.log(first)]
  rezl += [math.log(pl[(data2[x-1]/c,data2[x]/c,data2[x+1]/c,data1[x-1]/c)][data1[x]/c]) for x in range(1,size-1)]
  last = prior * pl[(data2[-2]/c,data2[-1]/c,0,data1[-2]/c)][data1[-1]/c] + (1 - prior) * pl[(data2[-2]/c,data2[-1]/c,1,data1[-2]/c)][data1[-1]/c]
  rezl += [math.log(last)]

  rezr = [math.log(first)]
  rezr += [math.log(pr[(data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c)][data2[x]/c]) for x in range(1,size-1)]
  last = prior * pr[(data1[-2]/c,data1[-1]/c,0,data2[-2]/c)][data2[-1]/c] + (1 - prior) * pr[(data1[-2]/c,data1[-1]/c,1,data2[-2]/c)][data2[-1]/c]
  rezr += [math.log(last)]

  rezl = math.fsum(rezl)
  rezr = math.fsum(rezr)
  deb = (rezl, rezr)

  return min(rezl,rezr)

def imgPredCostX(ra, rb, (wa,ha),(wb,hb), (pl,pr), prior, selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf
  data1 = ra[:ha]
  #data0 = ra[ha:2*ha]
  data2 = rb[-hb:]
  #if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
  #  return inf
  size = min(len(data1), len(data2))
  c = 255.0
  first = 1 - prior
  if data2[0] == 0:
    first = prior
  rezl = [math.log(first)]
  rezl += [math.log(pl[(data2[x-1]/c,data2[x]/c,data2[x+1]/c,data1[x-1]/c)][data1[x]/c]) for x in range(1,size-1)]
  last = prior * pl[(data2[-2]/c,data2[-1]/c,0,data1[-2]/c)][data1[-1]/c] + (1 - prior) * pl[(data2[-2]/c,data2[-1]/c,1,data1[-2]/c)][data1[-1]/c]
  rezl += [math.log(last)]

  rezr = [math.log(first)]
  rezr += [math.log(pr[(data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c)][data2[x]/c]) for x in range(1,size-1)]
  last = prior * pr[(data1[-2]/c,data1[-1]/c,0,data2[-2]/c)][data2[-1]/c] + (1 - prior) * pr[(data1[-2]/c,data1[-1]/c,1,data2[-2]/c)][data2[-1]/c]
  rezr += [math.log(last)]

  rezl = math.fsum(rezl)
  rezr = math.fsum(rezr)
  deb = (rezl, rezr)

  return min(rezl,rezr)

def calcPercentCost(page, px, py):
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgPerCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size, px)
      costY[(y, x)] = imgPerCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size, py)

  #print costX[((1,2),(0,2))]
  #print costX[((1,2),(1,3))]
  #assert False
  return costX, costY

def imgPerCostY(a, b, (wa,ha),(wb,hb), per, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = a[-wa:]
  data2 = b[:wb]

  size = min(len(data1), len(data2))
  c = 255.0
  rez = [math.log(per[(data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c)]) for x in range(1,size-1)]
  deb = (math.fsum(rez), rez)
  rez = math.fsum(rez)
  return rez

def imgPerCostX(ra, rb, (wa,ha),(wb,hb), per, selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf
  data1 = ra[:ha]
  data2 = rb[-hb:]

  size = min(len(data1), len(data2))
  c = 255.0
  rez = [math.log(per[(data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c)]) for x in range(1,size-1)]
  deb = (math.fsum(rez), rez)
  rez = math.fsum(rez)
  return rez

def calcNNCost(page):
  #for n in [(1, 1,1, 1,1, 1),(0, 0,0, 0,0, 0),(1,0, 1,0, 0,1),(0, 0,0, 0,1,0)]:
  #  print "Y", n, netY.activate(n)
  #  print "X", n, netX.activate(n)
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgNNCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size, dnx)
      costY[(y, x)] = imgNNCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size, dny)

  return costX, costY

def imgNNCostY(a, b, (wa,ha),(wb,hb), net, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = a[-wa:]
  data2 = b[:wb]
  #if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
  #  return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  c = 255.0
  #print filter(lambda x: x<=0, [(net.activate((data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c))[0]) for x in range(1,size-1)])
  rez += sum([math.log(net[(data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c)]) for x in range(1,size-1)])
  #self.totalCost += [rez]
  rez = rez / float(size-2)
  return rez

def imgNNCostX(ra, rb, (wa,ha),(wb,hb), net, selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf
  data1 = ra[:ha]
  data2 = rb[-hb:]
  #if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
  #  return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  c = 255.0
  #print filter(lambda x: x<=0, [(net.activate((data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c))[0]) for x in range(1,size-1)])

  rez += sum([math.log(net[(data1[x-1]/c,data1[x]/c,data1[x+1]/c,data2[x-1]/c,data2[x]/c,data2[x+1]/c)]) for x in range(1,size-1)])
  #self.totalCost += [rez]
  rez = rez / float(size-2)
  return rez

def calcRandCost(page, sx, sy): # random cost, used as benchmark
  pieces = page.states
  costX = {}
  costY = {}
  mult = max(sx,sy) * 100

  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = random.random() * 1.0/sy * mult
      costY[(y, x)] = random.random() * 1.0/sx * mult

  return costX, costY

def calcBitCost(page): # bit-bit difference 
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBitCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgBitCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def imgBitCostY(a, b, (wa,ha),(wb,hb), selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = a[-wa:]
  data2 = b[:wb]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(data1[x] - data2[x])/255.0 < delta else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def imgBitCostX(ra, rb, (wa,ha),(wb,hb), selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf
  data1 = ra[:ha]
  data2 = rb[-hb:]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(data1[x] - data2[x])/255.0 < delta else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def calcGausCost(page): # bit-gaussian difference, equivalent to smoothing
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgGausCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgGausCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def imgGausCostY(a, b, (wa,ha),(wb,hb), selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = a[-wa:]
  data2 = b[:wb]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(0.7*(data1[x] - data2[x]) + 0.1*(data1[x+1] - data2[x+1]) + 0.1*(data1[x-1] - data2[x-1]) + 0.05*(data1[x+2] - data2[x+2]) + 0.05*(data1[x-2] - data2[x-2]) )/255.0 < delta else 1 for x in range(2,size-2)])
  #self.totalCost += [rez]
  return rez

def imgGausCostX(ra, rb, (wa,ha),(wb,hb), selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf
  data1 = ra[:ha]
  data2 = rb[-hb:]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(0.7*(data1[x] - data2[x]) + 0.1*(data1[x+1] - data2[x+1]) + 0.1*(data1[x-1] - data2[x-1]) + 0.05*(data1[x+2] - data2[x+2]) + 0.05*(data1[x-2] - data2[x-2]) )/255.0 < delta else 1 for x in range(2,size-2)])
  #self.totalCost += [rez]
  return rez

def calcBlackRowCost(page): # bit-bit difference considering only black bits on rows and all bits on columns
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackBitCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgBitCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def calcBlackRowGausCost(page): # bit-bit difference considering only black bits on rows and all bits on columns
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackGausCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgGausCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def calcBlackBitCost(page): # bit-bit difference considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackBitCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgBlackBitCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def imgBlackBitCostY(a, b, (wa,ha), (wb,hb), selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = a[-wa:]
  data2 = b[:wb]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if data1[x] == 0 and data2[x] == 0  else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def imgBlackBitCostX(ra, rb, (wa,ha), (wb,hb), selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf

  data1 = ra[:ha]
  data2 = rb[-hb:]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if data1[x] == 0 and data2[x] == 0 else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def calcBlackGausCost(page): # bit-gaussian difference, equivalent to smoothing, but considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackGausCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgBlackGausCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def imgBlackGausCostY(a, b,(wa,ha), (wb,hb),  quant = 3, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf

  data1 = a[-wa:]
  data2 = b[:wb]
  if len(filter(lambda x: x == 0, data1)) < quant or len(filter(lambda x: x == 0, data2)) < quant:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  for x in range(1,size-1):
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
  #self.totalCost += [rez]
  return rez

def imgBlackGausCostX(ra, rb, (wa,ha), (wb,hb), quant = 3, selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf

  data1 = ra[:ha]
  data2 = rb[-hb:]
  if len(filter(lambda x: x == 0, data1)) < quant or len(filter(lambda x: x == 0, data2)) < quant:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  for x in range(1,size-1):
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
  #self.totalCost += [rez]
  return rez

def calcBlackBigGausCost(page): # bit-gaussian difference, equivalent to smoothing, but considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackBigGausCostX(page.rotDataPieces[y], page.rotDataPieces[x], pieces[y].size, pieces[x].size)
      costY[(y, x)] = imgBlackBigGausCostY(page.dataPieces[y], page.dataPieces[x], pieces[y].size, pieces[x].size)

  return costX, costY

def calcBigGaus(rez, size, data1, data2):
  for x in range(1,size-1):
    if data1[x] == 0:
      if data2[x-2] == 0:
        rez -= 0.5
      else:
        rez += 0.25
      if data2[x+2] == 0:
        rez -= 0.5
      else:
        rez += 0.25
      if data2[x-1] == 0:
        rez -= 1.5
      else:
        rez += 0.75
      if data2[x+1] == 0:
        rez -= 1.5
      else:
        rez += 0.75
      if data2[x] == 0:
        rez -= 3
      else:
        rez += 4
    if data2[x] == 0:
      if data1[x-2] == 0:
        rez -= 0.5
      else:
        rez += 0.25
      if data1[x+2] == 0:
        rez -= 0.5
      else:
        rez += 0.25
      if data1[x-1] == 0:
        rez -= 1.5
      else:
        rez += 0.75
      if data1[x+1] == 0:
        rez -= 1.5
      else:
        rez += 0.75
      if data1[x] == 0:
        rez -= 3
      else:
        rez += 4


  return rez

def imgBlackBigGausCostY(a, b,(wa,ha), (wb,hb),  selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf

  data1 = a[-wa:]
  data2 = b[:wb]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez = calcBigGaus(rez, size, data1, data2)
  #self.totalCost += [rez]
  return rez

def imgBlackBigGausCostX(ra, rb, (wa,ha), (wb,hb), selective = True, blank = None):
  if ra == rb:
    if not selective and ra == blank:
      return 0
    return inf

  data1 = ra[:ha]
  data2 = rb[-hb:]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez = calcBigGaus(rez, size, data1, data2)
  #self.totalCost += [rez]
  return rez
