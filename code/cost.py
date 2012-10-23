import random

delta = 0.1
inf = float("inf")

def picker(s, page, sx = None, sy = None):
  if s == "bit":
    return calcBitCost(page)
  elif s == "gaus":
    return calcGausCost(page)
  elif s == "blackBit":
    return calcBlackBitCost(page)
  elif s == "blackGaus":
    return calcBlackGausCost(page)
  elif s == "rand":
    return calcRandCost(page, sx, sy)
  elif s == "blackRow":
    return calcBlackRowCost(page)
  elif s == "blackRowGaus":
    return calcBlackRowGausCost(page)

def indivPicker(s, a,b, tp, selective = False, blank = None, sx = None, sy = None): #selective is a flag showing wether to give infinity of zero for stuff like blank on blank
  if s == "bit":
    if tp == "x":
      return imgBitCostX(a, b, selective, blank)
    elif tp == "y":
      return imgBitCostY(a, b, selective, blank)
  if s == "gaus":
    if tp == "x":
      return imgGausCostX(a, b, selective, blank)
    elif tp == "y":
      return imgGausCostY(a, b, selective, blank)
  if s == "blackBit":
    if tp == "x":
      return imgBlackBitCostX(a, b, selective, blank)
    elif tp == "y":
      return imgBlackBitCostY(a, b, selective, blank)
  if s == "blackGaus":
    if tp == "x":
      return imgBlackGausCostX(a, b, selective, blank)
    elif tp == "y":
      return imgBlackGausCostY(a, b, selective, blank)
  elif s == "rand":
    mult = max(sx,sy) * 100
    if tp == "x":
      return random.random() * 1.0/sy * mult
    elif tp == "y":
      return random.random() * 1.0/sx * mult

def evaluateCost(page, sx, sy): # calculate percent of edges whose best match given by the cost function is the true neighbour and ammount of error
  costX = page.costX
  costY = page.costY  
  bestX = {}
  bestY = {}
  for (k1,k2),v in costX.items():
    if k1 not in bestX or v < bestX[k1][1]:
      bestX[k1] = (k2, v)

  for (k1,k2),v in costY.items():
    if k1 not in bestY or v < bestY[k1][1]:
      bestY[k1] = (k2, v)

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
          correct += 1
        else:
          #print "XX", y,x
          error += abs(costX[((y,x),(cy,cx))] - bestX[(y,x)][1])

      cx = x
      cy = y + 1
      if cy < sy:
        count += 1
        if costY[((y,x),(cy,cx))] == bestY[(y,x)][1]:
          correct += 1
        else:
          #print "YY", y,x
          error += abs(costY[((y,x),(cy,cx))] - bestY[(y,x)][1])

  print float(correct) / count, float(error) / count
  #print sorted(bestX.items()), sorted(bestY.items())

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
      costX[(y, x)] = imgBitCostX(pieces[y], pieces[x])
      costY[(y, x)] = imgBitCostY(pieces[y], pieces[x])

  return costX, costY

def imgBitCostY(a, b, selective = True, blank = None):
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

def imgBitCostX(a, b, selective = True, blank = None):
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

def calcGausCost(page): # bit-gaussian difference, equivalent to smoothing
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgGausCostX(pieces[y], pieces[x])
      costY[(y, x)] = imgGausCostY(pieces[y], pieces[x])

  return costX, costY

def imgGausCostY(a, b, selective = True, blank = None):
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

def imgGausCostX(a, b, selective = True, blank = None):
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

def calcBlackRowCost(page): # bit-bit difference considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackBitCostX(pieces[y], pieces[x])
      costY[(y, x)] = imgBitCostY(pieces[y], pieces[x])

  return costX, costY

def calcBlackRowGausCost(page): # bit-bit difference considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackGausCostX(pieces[y], pieces[x])
      costY[(y, x)] = imgGausCostY(pieces[y], pieces[x])

  return costX, costY

def calcBlackBitCost(page): # bit-bit difference considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackBitCostX(pieces[y], pieces[x])
      costY[(y, x)] = imgBlackBitCostY(pieces[y], pieces[x])

  return costX, costY

def imgBlackBitCostY(a, b, selective = True, blank = None):
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

def imgBlackBitCostX(a, b, selective = True, blank = None):
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

def calcBlackGausCost(page): # bit-gaussian difference, equivalent to smoothing, but considering only black bits
  pieces = page.states
  costX = {}
  costY = {}
  for x in pieces.keys():
    for y in pieces.keys():
      costX[(y, x)] = imgBlackGausCostX(pieces[y], pieces[x])
      costY[(y, x)] = imgBlackGausCostY(pieces[y], pieces[x])

  return costX, costY

def imgBlackGausCostY(a, b, selective = True, blank = None):
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

def imgBlackGausCostX(a, b, selective = True, blank = None):
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
