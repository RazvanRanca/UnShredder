import heapq as hq
import cost
import numpy as np
import math
import operator
from decimal import *
import random

externalPenalty = 0
delta = 0.0001

def picker(s, page, ignoreWhites = False):
  if s == "g1D":
    return greedy1D(page)
  elif s == "prim":
    return prim(page)
  elif s == "kruskal":
    return kruskal(page, ignoreWhites)
  else:
    raise Exception("Unknown search method: " + s)

# TODO verify this.
def greedy1D(page): # takes best edges and adjoins them(starting with blankTopLeft), goes to next row on blank right side
  heap = page.heapX
  sortedQueue =  sorted(filter(lambda (score,(x,y)): x != y, heap))
  #print sortedQueue
  blankRight = page.blankRight # + page.blankPos

  blankTopLeft = list(set(page.blankLeft) & set(page.blankUp))
  rez = [] 
  row = 0
  seen = set()
  groups = {}
  gInd = 0
  cur = next(p for p in sortedQueue if p[1][0] in blankTopLeft)[1]
  rez.append(cur)
  seen.add(cur[0])
  seen.add(cur[1])
  heads = set()
  tails = set()
  heads.add(cur[0])
  tails.add(cur[1])
  tails.add(cur[0])
  #print sortedQueue
  sortedQueue = filter(lambda (score,(x,y)): x != cur[0] and y != cur[1] and x != cur[1] and y != cur[0], sortedQueue)
  #print sortedQueue
  groups[cur[0]] = gInd
  groups[cur[1]] = gInd
  gInd += 1
  while len(sortedQueue) > 0:
    cur = sortedQueue[0][1]
    rez.append(cur)
    seen.add(cur[0])
    seen.add(cur[1])
    heads.add(cur[0])
    tails.add(cur[1])
    sortedQueue = filter(lambda (score,(x,y)): x != cur[0] and y != cur[1] and x != cur[1] and y != cur[0], sortedQueue)
    groups[cur[0]] = gInd
    groups[cur[1]] = gInd
    gInd += 1

  edges = []
  for e1 in rez:
    for e2 in rez:
      if e1 != e2 and e2 != rez[0]:
        edges.append((e1[1],e2[0]))

  seen.add(page.blankPos) # we don't want to consider the blank piece
  remainingVertices = set(page.states.keys()) - seen
  #print rez
  #print remainingVertices
  for e1 in rez:
    for v in remainingVertices:
      edges.append((e1[1],v))
      edges.append((v,e1[0]))
      groups[v] = gInd
      gInd += 1

  #print edges
  heap, _ = page.extractHeap(edges)
  sortedQueue = filter(lambda (score,(x,y)): x not in heads and y not in tails, sorted(heap))
  #print heads, tails
  #print sortedQueue
  while len(sortedQueue) > 0:
    #print sortedQueue
    cur = sortedQueue[0][1]
    #print "ttt", cur, groups[cur[0]], groups[cur[1]]
    if groups[cur[0]] != groups[cur[1]]:
      rez.append(cur)
      heads.add(cur[0])
      tails.add(cur[1])
      #print cur
      sortedQueue = filter(lambda (score,(x,y)): x not in heads and y not in tails, sortedQueue)
      gInd = groups[cur[1]]
      for g in groups:
        if groups[g] == gInd:
          groups[g] = groups[cur[0]]
    else:
      sortedQueue = sortedQueue[1:]

  #print rez
  srez = [rez[0]]
  for i in range(0, len(rez)-1):
    #print srez
    srez.append(next(edge for edge in rez if (edge[0] == srez[i][1] and edge not in srez)))

  rez = [[]]
  row = 0
  count = 0
  for edge in srez:
    count += 1
    if edge[0] in blankRight and rez[row] != []:
      row += 1
      if count < len(srez):
        rez.append([])
    else:
      rez[row].append(edge)

  return rez

def prim(page): # adds best edge to graph already constructed, analog of Prim's agorithm.

  hX = page.heapX
  hY = page.heapY
  states = page.states.keys()
  found = set()
  posX = 0
  posY = 0
  cur = 0
  positions = {}
  grid = set()
  revGrid = {}
  accumEdges = []
  while len(states) > len(found) + 1:
    cX = 0
    cY = 0
    #print "---X---", sorted(hX, reverse=True)
    #print "---Y---", sorted(hY, reverse=True)
    
    hX = [(score,(f,s)) for (score,(f,s)) in hX if f != page.blankPos and s != page.blankPos]
    hY = [(score,(f,s)) for (score,(f,s)) in hY if f != page.blankPos and s != page.blankPos]
    maxX = min([x[0] for x in hX]) if hX != [] else None
    maxY = min([x[0] for x in hY]) if hY != [] else None

    bestX = random.choice(filter(lambda x: x[0] == maxX, hX)) if hX != [] else None
    bestY = random.choice(filter(lambda y: y[0] == maxY, hY)) if hY != [] else None


    #print "=============", bestX, bestY
    if bestY == None or (bestX != None and bestX[0] < bestY[0]):
      cur = bestX[1]
      accumEdges.append(("x",cur))
      if cur[1] in found:
        cX = -1
      else:
        cX = 1
    else:
      cur = bestY[1]
      accumEdges.append(("y",cur))
      if cur[1] in found:
        cY = -1
      else:
        cY = 1
    
    if len(positions) == 0:
      #cur = ((1,0),None) # CHEATING
      found.add(cur[0])
      positions[cur[0]] = (posY, posX)
      grid.add((posY, posX))
      revGrid[(posY, posX)] = cur[0]
    else:
      if cur[0] in found:
        posY, posX = positions[cur[0]]
      else:
        posY, posX = positions[cur[1]]
      posX += cX
      posY += cY


      found.add(cur[0])
      found.add(cur[1])

      if cur[0] not in positions:
        positions[cur[0]] = (posY, posX)
        grid.add((posY, posX))
        revGrid[(posY, posX)] = cur[0]
        #print cur[0], posX, posY

      if cur[1] not in positions:
        positions[cur[1]] = (posY, posX)
        grid.add((posY, posX))
        revGrid[(posY, posX)] = cur[1]
        #print cur[1], posX, posY

    hX = []
    hY = []
    #print accumEdges
    #print found
    #print positions
    #print grid

    nodeX1 = {}
    nodeX2 = {}
    nodeY1 = {}
    nodeY2 = {}
    for n in states:
      nodeX1[n] = {}
      nodeX2[n] = {}
      nodeY1[n] = {}
      nodeY2[n] = {}

    for f in found:
      for n in states:
        if n not in found:
          py,px = positions[f]
          if (py, px+1) not in grid:
            ay, ax = py, px+1
            nodeX1[f][n] = calcGridScore(ay, ax, n, grid, revGrid, page)

          if (py, px-1) not in grid:
            ay, ax = py, px-1
            nodeX2[f][n] = calcGridScore(ay, ax, n, grid, revGrid, page)

          if (py+1, px) not in grid:
            ay, ax = py+1, px
            nodeY1[f][n] = calcGridScore(ay, ax, n, grid, revGrid, page)

          if (py-1, px) not in grid:
            ay, ax = py-1, px
            nodeY2[f][n] = calcGridScore(ay, ax, n, grid, revGrid, page)

    #print "X1",nodeX1,'\n'
    #print "X2",nodeX2,'\n'
    #print "Y1",nodeY1,'\n'
    #print "Y2",nodeY2,'\n'
    """
    for n in states:

      #if lognormalize(nodeX1[n]) != fastLognormalize(nodeX1[n]):
      #  print "n", lognormalize(nodeX1[n]) 
      #  print "f", fastLognormalize(nodeX1[n])
      cX1 = lognormalize(dict([(x[0],x[1][0]) for x in nodeX1[n].items()]))
      cX2 = lognormalize(dict([(x[0],x[1][0]) for x in nodeX2[n].items()]))
      cY1 = lognormalize(dict([(x[0],x[1][0]) for x in nodeY1[n].items()]))
      cY2 = lognormalize(dict([(x[0],x[1][0]) for x in nodeY2[n].items()]))

      if not verifyNorm(cX1, cX2, cY1, cY2):
        print "goddammit"

      nodeX1[n] = dict([(k,(cX1[k],nodeX1[n][k][1])) for k in nodeX1[n]])
      nodeX2[n] = dict([(k,(cX2[k],nodeX2[n][k][1])) for k in nodeX2[n]])
      nodeY1[n] = dict([(k,(cY1[k],nodeY1[n][k][1])) for k in nodeY1[n]])
      nodeY2[n] = dict([(k,(cY2[k],nodeY2[n][k][1])) for k in nodeY2[n]])
    """

    for v1 in nodeX1:
      for v2 in nodeX1[v1]:
        #hX.append((nodeX1[v1][v2][0], (v1,v2)))
        hX.append((nodeX1[v1][v2][0]/nodeX1[v1][v2][1], (v1,v2)))

    for v1 in nodeX2:
      for v2 in nodeX2[v1]:
        #hX.append((nodeX2[v1][v2][0], (v2,v1)))
        hX.append((nodeX2[v1][v2][0]/nodeX2[v1][v2][1], (v2,v1)))

    for v1 in nodeY1:
      for v2 in nodeY1[v1]:
        #hY.append((nodeY1[v1][v2][0], (v1,v2)))
        hY.append((nodeY1[v1][v2][0]/nodeY1[v1][v2][1], (v1,v2)))

    for v1 in nodeY2:
      for v2 in nodeY2[v1]:
        #hY.append((nodeY2[v1][v2][0], (v2,v1)))
        hY.append((nodeY2[v1][v2][0]/nodeY2[v1][v2][1], (v2,v1)))

    #print nodeX1
    """
    for f in found:
      nX1 = {}
      nX2 = {}
      nY1 = {}
      nY2 = {}
      for n in states:
        if n not in found:
          py,px = positions[f]
          if (py, px+1) not in grid:
            ay, ax = py, px+1
            (score, count, _) = calcGridScore(ay, ax, n, grid, revGrid, page)
            if count == 0:
              assert score == cost.inf
              count = 1
            nX1[(f,n)] = (score, float(count))
            
          if (py, px-1) not in grid:
            ay, ax = py, px-1
            (score, count, _) = calcGridScore(ay, ax, n, grid, revGrid, page)
            if count == 0:
              assert score == cost.inf
              count = 1
            nX2[(n,f)] = (score, float(count))

          if (py+1, px) not in grid:
            ay, ax = py+1, px
            (score, count, _) = calcGridScore(ay, ax, n, grid, revGrid, page)
            if count == 0:
              assert score == cost.inf
              count = 1
            nY1[(f,n)] = (score, float(count))

          if (py-1, px) not in grid:
            ay, ax = py-1, px
            (score, count, _) = calcGridScore(ay, ax, n, grid, revGrid, page)
            if count == 0:
              assert score == cost.inf
              count = 1
            nY2[(n,f)] = (score, float(count))


      cX1 = flognormalize(dict([(x[0],x[1][0]) for x in nX1.items()]))
      cX2 = flognormalize(dict([(x[0],x[1][0]) for x in nX2.items()]))
      cY1 = flognormalize(dict([(x[0],x[1][0]) for x in nY1.items()]))
      cY2 = flognormalize(dict([(x[0],x[1][0]) for x in nY2.items()]))
      if not verifyNorm(cX1, cX2, cY1, cY2):
        print "goddammit"

      nX1 = dict([(k,(cX1[k],nX1[k][1])) for k in nX1])
      nX2 = dict([(k,(cX2[k],nX2[k][1])) for k in nX2])
      nY1 = dict([(k,(cY1[k],nY1[k][1])) for k in nY1])
      nY2 = dict([(k,(cY2[k],nY2[k][1])) for k in nY2])

      for v in nX1:
        hX.append((nX1[v], v))

      for v in nX2:
        hX.append((nX2[v], v))

      for v in nY1:
        hY.append((nY1[v], v))

      for v in nY2:
        hY.append((nY2[v], v))

    #hq.heapify(hX)
    #hq.heapify(hY)
    #page.vizPos(positions, fl="quuu.jpg")
    #raw_input("Press any key to continue...")
  """
    #page.vizPos(positions, fl="quuu.jpg")
    #raw_input("qqqq")

  assert len(set(positions.values())) == len(positions.values())
  #page.vizPos(positions, fl="qqqq" + str(page.sizeX) + ".jpg")
  #print positions
  #print accumEdges



  return positions, accumEdges

def verifyNorm(nodeX1, nodeX2, nodeY1={0:0}, nodeY2={0:0}):
  correct = True
  v1 = math.fsum(map(math.exp, nodeX1.values()))
  v2 = math.fsum(map(math.exp, nodeX2.values()))
  v3 = math.fsum(map(math.exp, nodeY1.values()))
  v4 = math.fsum(map(math.exp, nodeY2.values()))
  vs = {1:(v1,nodeX1),2:(v2,nodeX2),3:(v3,nodeY1),4:(v4,nodeY2)}
  for v in vs:
    if len(vs[v][1]) != 0 and abs(vs[v][0] - 1) > delta:
      print v, vs[v]
      correct = False

  return correct

def normalizeList(l, tup=False):
  if tup == False:
    total = float(sum(l))
    rez = [x/total for x in l]
  else:
    total = float(sum([x[0] for x in l]))
    rez = [(x[0]/total,x[1]) for x in l]
  return rez

def lognormalize(l):
  if len(l) == 0:
    return l
  x = [Decimal(str(q)) for q in l.values()]
  a = float(reduce(DefaultContext.add,[q.exp() for q in x]).ln())
  ret = dict([(q[0], q[1] - a) for q in l.items()])
  return ret

def flognormalize(l):
  if len(l) == 0:
    return l
  x = np.array([q for q in l.values()])
  a = np.logaddexp.reduce(x)
  ret = dict([(q[0], q[1] - a) for q in l.items()])
  return ret

#@profile
def calcGridScore(ay, ax, n, grid, revGrid, page):
  score = 0
  count = 0

  if (ay, ax+1) in grid:
    ts = page.costX[n, revGrid[(ay, ax+1)]]
    score += ts
    count += 1
  else:
    score += externalPenalty

  if (ay, ax-1) in grid:
    ts = page.costX[revGrid[(ay,ax-1)], n]
    score += ts
    count += 1
  else:
    score += externalPenalty

  if (ay+1, ax) in grid:
    ts = page.costY[n, revGrid[(ay+1, ax)]]
    score += ts
    count += 1
  else:
    score += externalPenalty

  if (ay-1, ax) in grid:
    ts = page.costY[revGrid[(ay-1,ax)], n]
    score += ts
    count += 1
  else:
    score += externalPenalty

  return score, count

#@profile
def kruskal(page, ignoreWhites = False): # constructs and merges forests of best nodes, analog of Kruskal's algorithm.
  hX = page.heapX
  hY = page.heapY
  states = page.states.keys()
  found = set()
  nodeForest = {}
  posX = 0
  posY = 0
  cur = 0
  positions = {}
  grid = {}
  revGrid = {}
  accumEdges = []
  numForests = 0
  firstOne = True

  while len(hX) > 0 or len(hY) > 0:
    if not firstOne:
      forest = None
      tp = None
      #print "---X---", [(math.exp(x[0]),x[1]) for x in sorted(hX, key=lambda x : x[1], reverse = True)]
      #print "---Y---", [(math.exp(x[0]),x[1]) for x in sorted(hY, key=lambda x : x[1], reverse = True)]

      hX = [(score,(f,s)) for (score,(f,s)) in hX if f != page.blankPos and s != page.blankPos]
      hY = [(score,(f,s)) for (score,(f,s)) in hY if f != page.blankPos and s != page.blankPos]
      if len(hX) == 0 and len(hY) == 0:
        break

      bestX = max(hX) if hX != [] else None
      bestY = max(hY) if hY != [] else None

      #print "+++++++++++++", bestX, bestY
      if bestY == None or (bestX != None and bestX[0] > bestY[0]):
        cur = bestX[1]
        accumEdges.append(("x",cur))
        tp = "x"
        #hY.append(bestY)
      else:
        cur = bestY[1]
        accumEdges.append(("y",cur))
        tp = "y"
        #hX.append(bestX)

      if cur[0] in found and cur[1] in found: # merge forests
        forest = nodeForest[cur[0]]
        posY, posX = positions[forest][cur[0]]
        if tp == "x":
          posX += 1
        elif tp == "y":
          posY += 1
        else:
          raise Exception ("Unrecognized edge type: " + str(tp))

        oldForest = nodeForest[cur[1]]
        oldY, oldX = positions[oldForest][cur[1]]
        dY = oldY - posY
        dX = oldX - posX
        #print cur[0], cur[1], forest, oldForest, revGrid, revGrid[oldForest].items()
        for ((pY, pX),node) in revGrid[oldForest].items():
          nodeForest[node] = forest
          grid[forest].add((pY-dY, pX-dX))
          revGrid[forest][(pY-dY, pX-dX)] = node
          positions[forest][node] = (pY-dY, pX-dX)

        del grid[oldForest]
        del revGrid[oldForest]
        del positions[oldForest]
        
      elif cur[0] in found: # add to forest
        forest = nodeForest[cur[0]]
        posY, posX = positions[forest][cur[0]]

        if tp == "x":
          posX += 1
        elif tp == "y":
          posY += 1
        else:
          raise Exception ("Unrecognized edge type: " + str(tp))

        positions[forest][cur[1]] = (posY, posX)
        found.add(cur[1])
        grid[forest].add((posY, posX))
        revGrid[forest][(posY, posX)] = cur[1]
        nodeForest[cur[1]] = forest

      elif cur[1] in found: # add to forest
        forest = nodeForest[cur[1]]
        posY, posX = positions[forest][cur[1]]
        if tp == "x":
          posX -= 1
        elif tp == "y":
          posY -= 1
        else:
          raise Exception ("Unrecognized edge type: " + str(tp))

        positions[forest][cur[0]] = (posY, posX)
        found.add(cur[0])
        grid[forest].add((posY, posX))
        revGrid[forest][(posY, posX)] = cur[0]
        nodeForest[cur[0]] = forest

      else: # make new forest
        posX = 0
        posY = 0
        posNX = 0
        posNY = 0
        if tp == "x":
          posNX = 1
        elif tp == "y":
          posNY = 1
        else:
          raise Exception ("Unrecognized edge type: " + str(tp))

        numForests += 1
        forest = numForests
        grid[forest] = set([(posY, posX),(posNY, posNX)])
        nodeForest[cur[0]] = forest
        nodeForest[cur[1]] = forest
        found.add(cur[0])
        found.add(cur[1])
        revGrid[forest] = {(posY, posX):cur[0], (posNY, posNX):cur[1]}
        positions[forest] = {cur[0]:(posY, posX), cur[1]:(posNY, posNX)}

    hX = []
    hY = []
    #print accumEdges
    #print found
    #print positions
    #print grid
    nodeX = {}
    nodeY = {}

    for n in states:
      nodeX[n] = {}
      nodeY[n] = {}

    #print "founD", found
    for i1 in range(len(states)):
      n1 = states[i1]
      if not ignoreWhites or n1 not in page.whites:
        for i2 in range(len(states)):
          n2 = states[i2]
          if i1 != i2 and (not ignoreWhites or n2 not in page.whites):
            if n1 in found and n2 in found and nodeForest[n1] != nodeForest[n2]: # possible merge
              f1 = nodeForest[n1]
              f2 = nodeForest[n2]
              p1Y, p1X = positions[f1][n1]
              p2Y, p2X = positions[f2][n2]

              np2Y, np2X = p1Y, p1X + 1
              dY, dX = p2Y - np2Y, p2X - np2X

              overlap = False
              infinite = True
              totalScore = 0
              totalCount = 0
              totalInfCount = 0
              for ((pY, pX),node) in revGrid[f2].items():
                ny, nx = pY-dY, pX-dX
                if (ny,nx) in grid[f1]:
                  overlap = True
                  break
                
                (score, count) = calcGridScore(ny, nx, node, grid[f1], revGrid[f1], page)
                if score != cost.inf:
                  infinite = False
                  totalScore += score
                  totalCount += count

              totalCount = 1
              if not overlap:
                if infinite:
                  nodeX[n1][n2] = (cost.inf, 1)
                else:
                  nodeX[n1][n2] = (totalScore,totalCount)

              np2Y, np2X = p1Y + 1, p1X
              dY, dX = p2Y - np2Y, p2X - np2X

              overlap = False
              infinite = True
              totalScore = 0
              totalCount = 0
              totalInfCount = 0
              for ((pY, pX),node) in revGrid[f2].items():
                ny, nx = pY-dY, pX-dX
                if (ny,nx) in grid[f1]:
                  overlap = True
                  break
                
                (score, count) = calcGridScore(ny, nx, node, grid[f1], revGrid[f1], page)
                if score != cost.inf:
                  infinite = False
                  totalScore += score
                  totalCount += count
 
              totalCount = 1
              if not overlap:
                if infinite:
                  nodeY[n1][n2] = (cost.inf, 1)
                else:
                  nodeY[n1][n2] = (totalScore,totalCount)

            if n1 in found and n2 not in found:
              forest = nodeForest[n1]
              py,px = positions[forest][n1]
              if (py, px+1) not in grid[forest]:
                ay, ax = py, px+1
                
                (score, count) = calcGridScore(ay, ax, n2, grid[forest], revGrid[forest], page)
                if count == 0:
                  assert score == cost.inf
                  nodeX[n1][n2] = (cost.inf, 1)
                else:
                  nodeX[n1][n2] = (score,count)

              if (py, px-1) not in grid[forest]:
                ay, ax = py, px-1
                (score, count) = calcGridScore(ay, ax, n2, grid[forest], revGrid[forest], page)
                if count == 0:
                  assert score == cost.inf
                  nodeX[n2][n1] = (cost.inf, 1)
                else:
                  nodeX[n2][n1] = (score,count)

              if (py+1, px) not in grid[forest]:
                ay, ax = py+1, px
                (score, count) = calcGridScore(ay, ax, n2, grid[forest], revGrid[forest], page)
                if count == 0:
                  assert score == cost.inf
                  nodeY[n1][n2] = (cost.inf, 1)
                else:
                  nodeY[n1][n2] = (score,count)

              if (py-1, px) not in grid[forest]:
                ay, ax = py-1, px
                (score, count) = calcGridScore(ay, ax, n2, grid[forest], revGrid[forest], page)
                if count == 0:
                  assert score == cost.inf
                  nodeY[n2][n1] = (cost.inf, 1)
                else:
                  nodeY[n2][n1] = (score,count)

            if n1 not in found and n2 not in found: # 2 outside edges
              nodeX[n1][n2] = (page.costX[(n1,n2)], 1)
              nodeY[n1][n2] = (page.costY[(n1,n2)], 1)

    #print "X",sorted(nodeX.items()),'\n'
    #print "Y",sorted(nodeY.items()),'\n'
    
    for n in states:

      #if lognormalize(nodeX1[n]) != fastLognormalize(nodeX1[n]):
      #  print "n", lognormalize(nodeX1[n]) 
      #  print "f", fastLognormalize(nodeX1[n])

      cX = lognormalize(dict([(x[0],x[1][0]) for x in nodeX[n].items()]))
      cY = lognormalize(dict([(x[0],x[1][0]) for x in nodeY[n].items()]))

      if not verifyNorm(cX, cY):
        print "goddammit"

      nodeX[n] = dict([(k,(cX[k],nodeX[n][k][1])) for k in nodeX[n]])
      nodeY[n] = dict([(k,(cY[k],nodeY[n][k][1])) for k in nodeY[n]])
    
    
    for v1 in nodeX:
      for v2 in nodeX[v1]:
        hX.append((nodeX[v1][v2][0], (v1,v2)))
        #hX.append((nodeX[v1][v2][0]/nodeX[v1][v2][1], (v1,v2)))

    for v1 in nodeY:
      for v2 in nodeY[v1]:
        hY.append((nodeY[v1][v2][0], (v1,v2)))
        #hY.append((nodeY[v1][v2][0]/nodeY[v1][v2][1], (v1,v2)))

    #if not firstOne:
      #page.vizPos({1:{(0,0):(0,1),(0,1):(0,1)}}, fl="quuu.jpg", multiple = True)
      #page.vizPos(positions, fl="quuu.jpg", multiple = True)
      #raw_input("Press any key to continue...")

    firstOne = False

  f = nodeForest[(0,0)]
  #print f, positions
  assert len(positions) == 1
  assert len(set(positions[f].values())) == len(positions[f].values())
  #page.vizPos(positions, fl="kruskal" + str(page.sizeX), multiple = True)
  #print positions
  #print accumEdges
  return positions[f], accumEdges
