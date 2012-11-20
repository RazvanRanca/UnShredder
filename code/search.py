import heapq as hq
import cost

externalPenalty = 0

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
  states = page.getAllStates()
  found = set()
  posX = 0
  posY = 0
  cur = 0
  positions = {}
  grid = set()
  revGrid = {}
  accumEdges = []
  while len(states) > len(found):
    cX = 0
    cY = 0
    #print "---X---", hX[:20]
    #print "---Y---", hY[:20]

    bestX = hq.heappop(hX) if hX != [] else None
    bestY = hq.heappop(hY) if hY != [] else None


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
    
    if len(positions) > 0:
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

    if len(positions) == 1:
      posX += cX
      posY += cY

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
    for f in found:
      for n in states:
        if n not in found:
          py,px = positions[f]
          if (py, px+1) not in grid:
            ay, ax = py, px+1
            (score, count, _) = calcGridScore(ay, ax, n, page, grid, revGrid)
            if count == 0:
              assert score == cost.inf
              count = 1
            hX.append((score/ float(count),(f,n)))
            
          if (py, px-1) not in grid:
            ay, ax = py, px-1
            (score, count, _) = calcGridScore(ay, ax, n, page, grid, revGrid)
            if count == 0:
              assert score == cost.inf
              count = 1
            hX.append((score/ float(count),(n,f)))

          if (py+1, px) not in grid:
            ay, ax = py+1, px
            (score, count, _) = calcGridScore(ay, ax, n, page, grid, revGrid)
            if count == 0:
              assert score == cost.inf
              count = 1
            hY.append((score/ float(count),(f,n)))

          if (py-1, px) not in grid:
            ay, ax = py-1, px
            (score, count, _) = calcGridScore(ay, ax, n, page, grid, revGrid)
            if count == 0:
              assert score == cost.inf
              count = 1
            hY.append((score/ float(count),(n,f)))

    hq.heapify(hX)
    hq.heapify(hY)

  assert len(set(positions.values())) == len(positions.values())

  #print positions
  #print accumEdges
  return positions, accumEdges

#@profile
def calcGridScore(ay, ax, n, page, grid, revGrid):
  score = 0
  infinite = True
  count = 0
  infCount = 0
  if (ay, ax+1) in grid:
    ts = page.costX[n, revGrid[(ay, ax+1)]]
    if ts != cost.inf:
      score += ts
      count += 1
      infinite = False 
    else:
      infCount += 1
  else:
    score += externalPenalty
  if (ay, ax-1) in grid:
    ts = page.costX[revGrid[(ay,ax-1)],n]
    if ts != cost.inf:
      score += ts
      count += 1
      infinite = False 
    else:
      infCount += 1
  else:
    score += externalPenalty
  if (ay+1, ax) in grid:
    ts = page.costY[n,revGrid[(ay+1, ax)]]
    if ts != cost.inf:
      score += ts
      count += 1
      infinite = False 
    else:
      infCount += 1
  else:
    score += externalPenalty
  if (ay-1, ax) in grid:
    ts = page.costY[revGrid[(ay-1,ax)],n]
    if ts != cost.inf:
      score += ts
      count += 1
      infinite = False 
    else:
      infCount += 1
  else:
    score += externalPenalty
  if infinite:
    score = cost.inf

  return score, count, infCount

#@profile
def kruskal(page, ignoreWhites = False): # constructs and merges forests of best nodes, analog of Kruskal's algorithm.
  hX = page.heapX
  hY = page.heapY
  states = page.getAllStates()
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

  while len(hX) > 0 or len(hY) > 0:
    forest = None
    tp = None
    #print "---X---", hX[:20]
    #print "---Y---", hY[:20]

    bestX = hq.heappop(hX) if hX != [] else None
    bestY = hq.heappop(hY) if hY != [] else None

    #print "+++++++++++++", bestX, bestY
    if bestY == None or (bestX != None and bestX[0] < bestY[0]):
      cur = bestX[1]
      accumEdges.append(("x",cur))
      tp = "x"
      hq.heappush(hY, bestY)
    else:
      cur = bestY[1]
      accumEdges.append(("y",cur))
      tp = "y"
      hq.heappush(hX, bestX)

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
    for i1 in range(len(states)):
      n1 = states[i1]
      if not ignoreWhites or n1 not in page.whites:
        for i2 in range(len(states)):
          n2 = states[i2]
          if not ignoreWhites or n2 not in page.whites:
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
                
                (score, count, infCount) = calcGridScore(ny, nx, node, page, grid[f1], revGrid[f1])
                if score != cost.inf:
                  infinite = False
                  totalScore += score
                  totalCount += count
                else:
                  totalInfCount += infCount

              if not overlap:
                if infinite:
                  hX.append(((cost.inf, -1*totalInfCount),(n1,n2)))
                else:
                  hX.append((totalScore/float(totalCount),(n1,n2)))

              np2Y, np2X = p1Y + 1, p1X
              dY, dX = p2Y - np2Y, p2X - np2X

              overlap = False
              infinite = True
              totalScore = 0
              totalCount = 0
              for ((pY, pX),node) in revGrid[f2].items():
                ny, nx = pY-dY, pX-dX
                if (ny,nx) in grid[f1]:
                  overlap = True
                  break
                
                (score, count, infCount) = calcGridScore(ny, nx, node, page, grid[f1], revGrid[f1])
                if score != cost.inf:
                  infinite = False
                  totalScore += score
                  totalCount += count
                else:
                  totalInfCount += infCount

              if not overlap:
                if infinite:
                  hY.append(((cost.inf, -1*totalInfCount),(n1,n2)))
                else:
                  hY.append((totalScore/float(totalCount),(n1,n2)))

            if n1 in found and n2 not in found:
              forest = nodeForest[n1]
              py,px = positions[forest][n1]
              if (py, px+1) not in grid[forest]:
                ay, ax = py, px+1
                (score, count, infCount) = calcGridScore(ay, ax, n2, page, grid[forest], revGrid[forest])
                if count == 0:
                  assert score == cost.inf
                  hX.append(((cost.inf, -1*infCount),(n1,n2)))
                else:
                  hX.append((score/ float(count),(n1,n2)))
                
              if (py, px-1) not in grid[forest]:
                ay, ax = py, px-1
                (score, count, infCount) = calcGridScore(ay, ax, n2, page, grid[forest], revGrid[forest])
                if count == 0:
                  assert score == cost.inf
                  hX.append(((cost.inf, -1*infCount),(n2,n1)))
                else:
                  hX.append((score/ float(count),(n2,n1)))

              if (py+1, px) not in grid[forest]:
                ay, ax = py+1, px
                (score, count, infCount) = calcGridScore(ay, ax, n2, page, grid[forest], revGrid[forest])
                if count == 0:
                  assert score == cost.inf
                  hY.append(((cost.inf, -1*infCount),(n1,n2)))
                else:
                  hY.append((score/ float(count),(n1,n2)))

              if (py-1, px) not in grid[forest]:
                ay, ax = py-1, px
                (score, count, infCount) = calcGridScore(ay, ax, n2, page, grid[forest], revGrid[forest])
                if count == 0:
                  assert score == cost.inf
                  hY.append(((cost.inf, -1*infCount),(n2,n1)))
                else:
                  hY.append((score/ float(count),(n2,n1)))

            if n1 not in found and n2 not in found: # 2 outside edges
              hX.append((page.costX[(n1,n2)],(n1,n2)))
              hY.append((page.costY[(n1,n2)],(n1,n2)))

    hq.heapify(hX)
    hq.heapify(hY)
      
    #page.vizPos(positions, fl="quuu", multiple = True)
    #raw_input("Press any key to continue...")

  f = nodeForest[(0,0)]
  #print f, positions
  assert len(positions) == 1
  assert len(set(positions[f].values())) == len(positions[f].values())

  #print positions
  #print accumEdges
  return positions[f], accumEdges
