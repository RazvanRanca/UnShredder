import heapq as hq

def picker(s, page):
  if s == "g1D":
    return greedy1D(page)
  elif s == "prim":
    return prim(page)
  else:
    raise Exception("Unknown search method: " + s)

# TODO verify this.
def greedy1D(page): # takes best edges and adjoins them(starting with blankTopLeft), goes to next row on blank right side
  heap = page.heapX
  sortedQueue =  sorted(filter(lambda (score,(x,y)): x != y, heap))
  #print sortedQueue
  blankRight = page.blankRight + page.blankPos

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
  accumEdges = []
  while len(states) > len(found):
    cX = 0
    cY = 0
    bestX = hq.heappop(hX) if hX != [] else None
    bestY = hq.heappop(hY) if hY != [] else None
    #print bestX, bestY
    if bestY == None or (bestX != None and bestX[0] <= bestY[0]):
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
      #print cur[0], posX, posY

    if len(positions) == 1:
      posX += cX
      posY += cY

    if cur[1] not in positions:
      positions[cur[1]] = (posY, posX)
      grid.add((posY, posX))
      #print cur[1], posX, posY

    edgesX = []
    edgesY = []
    #print accumEdges
    #print found
    #print positions
    #print grid
    for f in found:
      for n in states:
        if n not in found:
          py,px = positions[f]
          if (py, px+1) not in grid:
            edgesX.append((f,n))
          elif (py, px-1) not in grid:
            edgesX.append((n,f))
          if (py+1, px) not in grid:
            edgesY.append((f,n))
          elif (py-1, px) not in grid:
            edgesY.append((n,f))

    #print "edgesX: ", edgesX
    #print edgesY
    hX,hY = page.extractHeap(edgesX, edgesY)

  assert len(set(positions.values())) == len(positions.values())

  #print positions
  #print accumEdges
  return positions, accumEdges
