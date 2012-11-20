import random
import cost

### Slow/old implementation of all cost functions. Used to check correctness of faster implementation.

delta = 0.1
inf = 1000 # float("inf")

def checkAll(page):
  correct = True

  if slowCalcBitCost(page) == cost.calcBitCost(page):
    print "bit OK"
  else:
    print "bit WRONG"
    correct = False
  if slowCalcGausCost(page) == cost.calcGausCost(page):
    print "gaus OK"
  else:
    print "gaus WRONG"
    correct = False
  if slowCalcBlackBitCost(page) == cost.calcBlackBitCost(page):
    print "blackBit OK"
  else:
    print "blackBit WRONG"
    correct = False
  if slowCalcBlackGausCost(page) == cost.calcBlackGausCost(page): 
    print "blackGaus OK"
  else:
    print "blackGaus WRONG"
    correct = False
  if slowCalcBlackRowCost(page) == cost.calcBlackRowCost(page):
    print "blackRow OK"
  else:
    print "blackRow WRONG"
    correct = False
  if slowCalcBlackRowGausCost(page) == cost.calcBlackRowGausCost(page):
    print "blackRowGaus OK"
  else:
    print "blackRowGaus WRONG"
    correct = False

  return correct  

def picker(s, page, sx = None, sy = None):
  if s == "bit":
    return slowCalcBitCost(page) == cost.calcBitCost(page)
  elif s == "gaus":
    return slowCalcGausCost(page) == cost.calcGausCost(page)
  elif s == "blackBit":
    return slowCalcBlackBitCost(page) == cost.calcBlackBitCost(page)
  elif s == "blackGaus":
    return slowCalcBlackGausCost(page) == cost.calcBlackGausCost(page)
  elif s == "blackRow":
    return slowCalcBlackRowCost(page) == cost.calcBlackRowCost(page)
  elif s == "blackRowGaus":
    return slowCalcBlackRowGausCost(page) == cost.calcBlackRowGausCost(page)

def indivPicker(s, a,b, tp, selective = False, blank = None, sx = None, sy = None): #selective is a flag showing wether to give infinity of zero for stuff like blank on blank
  if s == "bit":
    if tp == "x":
      return slowImgBitCostX(a, b, selective, blank) == cost.imgBitCostX(a, b, selective, blank)
    elif tp == "y":
      return slowImgBitCostY(a, b, selective, blank) == cost.imgBitCostY(a, b, selective, blank)
  elif s == "gaus":
    if tp == "x":
      return slowImgGausCostX(a, b, selective, blank) == cost.imgGausCostX(a, b, selective, blank)
    elif tp == "y":
      return slowImgGausCostY(a, b, selective, blank) == cost.imgGausCostY(a, b, selective, blank)
  elif s == "blackBit":
    if tp == "x":
      return slowImgBlackBitCostX(a, b, selective, blank) == cost.imgBlackBitCostX(a, b, selective, blank)
    elif tp == "y":
      return slowImgBlackBitCostY(a, b, selective, blank) == cost.imgBlackBitCostY(a, b, selective, blank)
  elif s == "blackGaus":
    if tp == "x":
      return slowImgBlackGausCostX(a, b, selective, blank) == cost.imgBlackGausCostX(a, b, selective, blank)
    elif tp == "y":
      return slowImgBlackGausCostY(a, b, selective, blank) == cost.imgBlackGausCostY(a, b, selective, blank)
  elif s == "blackRow":
    if tp == "x":
      return slowImgBlackBitCostX(a, b, selective, blank) == cost.imgBlackBitCostX(a, b, selective, blank)
    elif tp == "y":
      return slowImgBitCostY(a, b, selective, blank) == cost.imgBlackBitCostY(a, b, selective, blank)
  elif s == "blackRowGaus":
    if tp == "x":
      return slowImgBlackGausCostX(a, b, selective, blank) == cost.imgBlackGausCostY(a, b, selective, blank)
    elif tp == "y":
      return slowImgGausCostY(a, b, selective, blank) == cost.imgBlackGausCostY(a, b, selective, blank)

def evaluateCost(page, sx, sy): # calculate percent of edges whose best match given by the cost function is the true neighbour and ammount of error
  costX = page.costX
  costY = page.costY  
  bestX = {}
  bestY = {}
  countX = 0.0
  countY = 0.0
  for (k1,k2),v in costX.items():
    if k1 not in bestX or v < bestX[k1][1]:
      bestX[k1] = (k2, v)
      countX = 1
    elif v == bestX[k1][1]:
      countX += 1

  for (k1,k2),v in costY.items():
    if k1 not in bestY or v < bestY[k1][1]:
      bestY[k1] = (k2, v)
      countY = 1
    elif v == bestY[k1][1]:
      countY += 1

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
          correct += 1.0/countX
        else:
          #print "XX", y,x
          error += abs(costX[((y,x),(cy,cx))] - bestX[(y,x)][1])

      cx = x
      cy = y + 1
      if cy < sy:
        count += 1
        if costY[((y,x),(cy,cx))] == bestY[(y,x)][1]:
          correct += 1.0/countY
        else:
          #print "YY", y,x
          error += abs(costY[((y,x),(cy,cx))] - bestY[(y,x)][1])

  return float(correct) / count, float(error) / count
  #print sorted(bestX.items()), sorted(bestY.items())

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

def slowCalcBitCost(page): # bit-bit difference
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = slowImgBitCostX(pieces[y], pieces[x])
      costY[(y, x)] = slowImgBitCostY(pieces[y], pieces[x])

  return costX, costY

def slowImgBitCostY(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = list(a.getdata())[-a.size[0]:]
  data2 = list(b.getdata())[:b.size[0]]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(data1[x] - data2[x])/255.0 < delta else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def slowImgBitCostX(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = list(a.rotate(90).getdata())[:a.size[1]]
  data2 = list(b.rotate(90).getdata())[-b.size[1]:]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(data1[x] - data2[x])/255.0 < delta else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def slowCalcGausCost(page): # bit-gaussian difference, equivalent to smoothing
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = slowImgGausCostX(pieces[y], pieces[x])
      costY[(y, x)] = slowImgGausCostY(pieces[y], pieces[x])

  return costX, costY

def slowImgGausCostY(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = list(a.getdata())[-a.size[0]:]
  data2 = list(b.getdata())[:b.size[0]]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(0.7*(data1[x] - data2[x]) + 0.1*(data1[x+1] - data2[x+1]) + 0.1*(data1[x-1] - data2[x-1]) + 0.05*(data1[x+2] - data2[x+2]) + 0.05*(data1[x-2] - data2[x-2]) )/255.0 < delta else 1 for x in range(2,size-2)])
  #self.totalCost += [rez]
  return rez

def slowImgGausCostX(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = list(a.rotate(90).getdata())[:a.size[1]]
  data2 = list(b.rotate(90).getdata())[-b.size[1]:]
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if abs(0.7*(data1[x] - data2[x]) + 0.1*(data1[x+1] - data2[x+1]) + 0.1*(data1[x-1] - data2[x-1]) + 0.05*(data1[x+2] - data2[x+2]) + 0.05*(data1[x-2] - data2[x-2]) )/255.0 < delta else 1 for x in range(2,size-2)])
  #self.totalCost += [rez]
  return rez

def slowCalcBlackRowCost(page): # bit-bit difference considering only black bits on rows and all bits on columns
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = slowImgBlackBitCostX(pieces[y], pieces[x])
      costY[(y, x)] = slowImgBitCostY(pieces[y], pieces[x])

  return costX, costY

def slowCalcBlackRowGausCost(page): # bit-bit difference considering only black bits on rows and all bits on columns
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = slowImgBlackGausCostX(pieces[y], pieces[x])
      costY[(y, x)] = slowImgGausCostY(pieces[y], pieces[x])

  return costX, costY

def slowCalcBlackBitCost(page): # bit-bit difference considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = slowImgBlackBitCostX(pieces[y], pieces[x])
      costY[(y, x)] = slowImgBlackBitCostY(pieces[y], pieces[x])

  return costX, costY


def slowImgBlackBitCostY(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf
  data1 = list(a.getdata())[-a.size[0]:]
  data2 = list(b.getdata())[:b.size[0]]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if data1[x] == 0 and data2[x] == 0  else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def slowImgBlackBitCostX(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf

  data1 = list(a.rotate(90).getdata())[:a.size[1]]
  data2 = list(b.rotate(90).getdata())[-b.size[1]:]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
    return inf
  size = min(len(data1), len(data2))
  rez = max(len(data1), len(data2)) - size
  rez += sum([0 if data1[x] == 0 and data2[x] == 0 else 1 for x in range(size)])
  #self.totalCost += [rez]
  return rez

def slowCalcBlackGausCost(page): # bit-gaussian difference, equivalent to smoothing, but considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = slowImgBlackGausCostX(pieces[y], pieces[x])
      costY[(y, x)] = slowImgBlackGausCostY(pieces[y], pieces[x])

  return costX, costY

def slowImgBlackGausCostY(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf

  data1 = list(a.getdata())[-a.size[0]:]
  data2 = list(b.getdata())[:b.size[0]]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
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

def slowImgBlackGausCostX(a, b, selective = True, blank = None):
  if a == b:
    if not selective and a == blank:
      return 0
    return inf

  data1 = list(a.rotate(90).getdata())[:a.size[1]]
  data2 = list(b.rotate(90).getdata())[-b.size[1]:]
  if len(filter(lambda x: x == 0, data1)) < 3 or len(filter(lambda x: x == 0, data2)) < 3:
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
