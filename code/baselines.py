import sys
from matplotlib import pyplot as plt
import search
import cost
import verifCost
import pages
import time
from PIL import Image
import random
import numpy as np

def genEdges(sx, sy): # edges of page starting at (0,0)
  xedges = set()
  yedges = set()
  for r in range(sy-1):
    for c in range(sx-1):
      yedges.add(((r,c),(r+1,c)))
      xedges.add(((r,c),(r,c+1)))
    yedges.add(((r,sx-1),(r+1,sx-1)))

  for c in range(sx-1):
    xedges.add(((sy-1,c),(sy-1,c+1)))

  return xedges, yedges

def simPrim(f, sx, sy, rand = False): # simulate prim with failure prob f
  states = []
  for r in range(sy):
    for c in range(sx):
      states.append((r,c))

  edgesX = []
  edgesY = []
  grid = {}
  rgrid = {}
  avail = set()
  nr = [0,0]
  nc = [1,-1]
  #nr = [0,0,1,-1]
  #nc = [1,-1,0,0]
  start = random.choice(states)
  states.remove(start)
  grid[start] = (0,0)
  #print "start", start, "on (0, 0)"
  rgrid[(0,0)] = start
  cr, cc = (0,0)
  for i in range(len(nr)):
    avail.add((cr + nr[i], cc + nc[i]))

  while len(states) > 0:
    nextState = None
    if random.random() < f:
      if rand:
        next = random.choice(list(avail))
        nextState = random.choice(states)
      else:
        random.shuffle(list(avail))
        
        for next in avail:
          cr, cc = next

          neigh = []
          for i in range(len(nr)):
            v = (cr + nr[i], cc + nc[i])
            if v in rgrid:
              neigh.append((rgrid[v][0] - nr[i], rgrid[v][1] - nc[i]))
          
          nstates = [x for x in states if x not in neigh]
          if nstates == []:
            continue
          nextState = random.choice(nstates)
          #print nextState, next, neigh
          break

    else:
      #print "non-random"
      favail = [(r,c) for (r,c) in avail if r >= -start[0] and c >= -start[1] and r < sy-start[0] and c < sx-start[1]]
      next = random.choice(favail)
      cr, cc = next

      for i in range(len(nr)):
        v = (cr + nr[i], cc + nc[i])
        if v in rgrid:
          rnr, rnc = v
          break

      ar, ac = (cr - rnr, cc - rnc)
      csr, csc = rgrid[(rnr,rnc)]
      corState = (csr + ar, csc + ac)
      if corState in states:      
        nextState = corState
        #print "determ", nextState, "on", next
      else:
        nextState = random.choice(states)
        #print "determ", corState, "not avail, so",nextState, "on", next

    avail.remove(next)
    if nextState == None:
      nextState = random.choice(states)
    states.remove(nextState)
    grid[nextState] = next
    rgrid[next] = nextState

    for i in range(len(nr)):
      v = (next[0] + nr[i], next[1] + nc[i])
      if v in rgrid:
        fr = rgrid[min(v,next)]
        to = rgrid[max(v,next)]
        if nr[i] != 0:
          edgesY.append((fr,to))
        else:
          edgesX.append((fr,to))
      else:
        avail.add(v)

  return edgesX, edgesY

def simKruskal(f, sx, sy, rand = False):
  forest = {}
  revForest = {}
  count = 0
  avail = {}
  grid = {}
  revGrid = {}
  neighr = [0,0]
  neighc = [1,-1]
  #neighr = [0,0,1,-1]
  #neighc = [1,-1,0,0]
  for r in range(sy):
    for c in range(sx):
      forest[count] = [(r,c)]
      revForest[(r,c)] = count
      revGrid[count] = {(0,0) : (r,c)}
      grid[count] = {(r,c) : (0,0)}
      avail[count] = set([(-1,0),(1,0),(0,-1),(0,1)])
      count += 1

  edgesX = []
  edgesY = []

  while len(forest) > 1:
    matches = {}
    seen = set()
    for f1 in forest:
      for f2 in forest:
        if f1 != f2 and (f1,f2) not in seen:
          seen.add((f1,f2))
          seen.add((f2,f1))
          for a1 in avail[f1]:
            for a2 in revGrid[f2]:
              ofr, ofc = (a1[0] - a2[0], a1[1] - a2[1])
              overlap = False
              for n in revGrid[f2]:
                nr = n[0] + ofr
                nc = n[1] + ofc
                if (nr,nc) in revGrid[f1]:
                  overlap = True
                  break
              if not overlap:
                
                corEdges = 0
                totEdges = 0
                for n in revGrid[f2]:
                  nr = n[0] + ofr
                  nc = n[1] + ofc
                  for i in range(len(neighr)):
                    nnr = nr + neighr[i]
                    nnc = nc + neighc[i]
                    if (nnr,nnc) in revGrid[f1]:
                      totEdges += 1
                      f1Neigh = revGrid[f1][(nnr,nnc)]
                      corNeigh = (f1Neigh[0] - neighr[i], f1Neigh[1] - neighc[i])
                      if corNeigh == revGrid[f2][n]:
                        #print f1, f2, a1, a2, revGrid[f1], revGrid[f2], ofr, ofc, nr, nc, f1Neigh, corNeigh
                        corEdges += 1
                if totEdges > 0:
                  matches[(f1,f2,(ofr, ofc))] = float(corEdges) / totEdges
    #print [(revGrid[f1][(0,0)], revGrid[f2][(0,0)], offset, score) for ((f1,f2,offset),score) in sorted(matches.items(), key=lambda x: x[1], reverse = True)]
    #return
    
    if random.random() < f:
      if rand:
        next = random.choice(matches.keys())
      else:
        minVal = min(matches.values())
        worst = filter(lambda (x,y): y == minVal, matches.items())
        next = random.choice(worst)[0]
    else:
        maxVal = max(matches.values())
        best = filter(lambda (x,y): y == maxVal, matches.items())
        #print [(revGrid[f1][(0,0)], revGrid[f2][(0,0)], offset, score) for ((f1,f2,offset),score) in best]
        next = random.choice(best)[0]

    (f1, f2, (ofr, ofc)) = next
    nodes =  revGrid[f2].items()
    for (pos, val) in nodes:
      npos = (pos[0] + ofr, pos[1] + ofc)
      forest[f1].append(val)
      revForest[val] = f1
      revGrid[f1][npos] = val
      grid[val] = npos

    for pos in revGrid[f1]:
      for i in range(len(neighr)):
        npos = (pos[0] + neighr[i], pos[1] + neighc[i])
        if npos not in revGrid[f1]:
          avail[f1].add(npos)

    del avail[f2]
    del grid[f2]
    del revGrid[f2]
    del forest[f2]

  xedges = set()
  yedges = set()
  f = revForest[(0,0)]
  for p1 in revGrid[f]:
    for i in range(len(neighr)):
      p2 = (p1[0] + neighr[i], p1[1] + neighc[i])
      if p2 in revGrid[f]:
        fr = revGrid[f][min(p1,p2)]
        to = revGrid[f][max(p1,p2)]
        if i < 2:
          xedges.add((fr,to))
        else:
          yedges.add((fr,to))

  return xedges, yedges


def simError(ex1,ey1,ex2,ey2):
  #print "cor length", len(ex1), len(ey1)
  #print "sim length", len(ex2), len(ey2)

  cx = set(ex1).intersection(set(ex2))
  cy = set(ey1).intersection(set(ey2))
 
  if len(ex1) > 0:
    ex = float(len(cx)) / len(ex1)
  else:
    ex = 1.0

  if len(ey1) > 0:
    ey = float(len(cy)) / len(ey1) 
  else:
    ey = 1.0
  et = (float(len(cx)) + float(len(cy))) / (len(ex1) + len(ey1)) 
  return ex, ey, et

def getPrediction(ims):
  predXl = {}
  predYl = {}
  predXr = {}
  predYr = {}
  prior = 0
  for im in ims:
    w,h = im.size
    ld = list(im.getdata())
    
    prior = ld.count(0) / float(len(ld))
    #print w,h, len(ld)
    bd = []
    for r in range(h):
      bd.append([])
      for c in range(w):
        bd[r].append(ld[r*w + c])

    div = 255.0
    for r in range(1,h-1):
      for c in range(1,w):
        nl = (bd[r-1][c]/div,bd[r][c]/div,bd[r+1][c]/div,bd[r-1][c-1]/div)#,bd[r][c]/div,bd[r+1][c]/div,bd[r-1][c+1]/div)
        vl = bd[r][c-1]/div
        try:
          predXl[nl][vl] += 1
        except:
          predXl[nl] = {vl : 2.0, 1-vl : 1.0}

        nr = (bd[r-1][c-1]/div,bd[r][c-1]/div,bd[r+1][c-1]/div,bd[r-1][c]/div)#,bd[r][c]/div,bd[r+1][c]/div,bd[r-1][c+1]/div)
        vr = bd[r][c]/div
        try:
          predXr[nr][vr] += 1
        except:
          predXr[nr] = {vr : 2.0, 1-vr : 1.0}

    for r in range(1,h):
      for c in range(1,w-1):
        nl = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div,bd[r-1][c-1]/div)#,bd[r][c]/div,bd[r][c+1]/div,bd[r+1][c-1]/div)
        vl = bd[r-1][c]/div
        try:
          predYl[nl][vl] += 1
        except:
          predYl[nl] = {vl : 2.0, 1-vl : 1.0}

        nr = (bd[r-1][c-1]/div,bd[r-1][c]/div,bd[r-1][c+1]/div,bd[r][c-1]/div)#,bd[r][c]/div,bd[r][c+1]/div,bd[r+1][c-1]/div)
        vr = bd[r][c]/div
        try:
          predYr[nr][vr] += 1
        except:
          predYr[nr] = {vr : 2.0, 1-vr : 1.0}

  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          n = tuple(map(float,(i1,i2,i3,i4)))
          if n not in predXl:
            print n
            predXl[n] = {0.0 : prior, 1.0 : (1-prior)}
          if n not in predYl:
            print n
            predYl[n] = {0.0 : prior, 1.0 : (1-prior)}

          if n not in predXr:
            print n
            predXr[n] = {0.0 : prior, 1.0 : (1-prior)}
          if n not in predYr:
            print n
            predYr[n] = {0.0 : prior, 1.0 : (1-prior)}

          predXl[n] = normalize(predXl[n])
          predYl[n] = normalize(predYl[n])
          predXr[n] = normalize(predXr[n])
          predYr[n] = normalize(predYr[n])

  return prior, predXl, predYl, predXr, predYr

def getPercents(ims):
  perX = {}
  perY = {}
  threeX = {}
  threeY = {}
  for im in ims:
    w,h = im.size
    ld = list(im.getdata())
    #print w,h, len(ld)
    bd = []
    for r in range(h):
      bd.append([])
      for c in range(w):
        bd[r].append(ld[r*w + c])

    div = 255.0
    for r in range(1,h-1):
      for c in range(w-1):
        n = (bd[r-1][c]/div,bd[r][c]/div,bd[r+1][c]/div,bd[r-1][c+1]/div,bd[r][c+1]/div,bd[r+1][c+1]/div)
        t1 = (bd[r-1][c]/div,bd[r][c]/div,bd[r+1][c]/div)
        t2 = (bd[r-1][c+1]/div,bd[r][c+1]/div,bd[r+1][c+1]/div)
        try:
          perX[n] += 1
        except:
          perX[n] = 1.0
        try:
          threeX[t1] += 1
        except:
          threeX[t1] = 1.0
        try:
          threeX[t2] += 1
        except:
          threeX[t2] = 1.0

    for r in range(h-1):
      for c in range(1,w-1):
        n = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div,bd[r+1][c-1]/div,bd[r+1][c]/div,bd[r+1][c+1]/div)
        t1 = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div)
        t2 = (bd[r+1][c-1]/div,bd[r+1][c]/div,bd[r+1][c+1]/div)
        try:
          perY[n] += 1
        except:
          perY[n] = 1.0
        try:
          threeY[t1] += 1
        except:
          threeY[t1] = 1.0
        try:
          threeY[t2] += 1
        except:
          threeY[t2] = 1.0

  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        t = (i1,i2,i3)
        if t not in threeX:
          print t
          threeX[t] = 0.1
        if t not in threeY:
          print t
          threeY[t] = 0.1
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              if n not in perX:
                print n
                perX[n] = 0.1
              if n not in perY:
                print n
                perY[n] = 0.1

  perX = normalize(perX)
  perY = normalize(perY)
  threeX = normalize(threeX)
  threeY = normalize(threeY)

  surX = {}
  surY = {}
  for n in perX:
    t1 = (n[0],n[1],n[2])
    t2 = (n[3],n[4],n[5])
    nt = (n[3],n[4],n[5],n[0],n[1],n[2])
    surX[n] = perX[n] / threeX[t1]

  for n in perY:
    t1 = (n[0],n[1],n[2])
    t2 = (n[3],n[4],n[5])
    nt = (n[3],n[4],n[5],n[0],n[1],n[2])
    surY[n] = perY[n] / threeY[t1]

  """
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        totalX = 0
        totalY = 0
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i4,i5,i6,i1,i2,i3)
              totalX += surX[n]
              totalY += surY[n]
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i4,i5,i6,i1,i2,i3)
              surX[n] /= totalX
              surY[n] /= totalY
  """

  #print "sX"
  #print '\n'.join(map(str,sorted(surX.items(), key=lambda x:x[0])))
  #print "sY"
  #print '\n'.join(map(str,sorted(surY.items(), key=lambda x:x[0])))
  return surX, surY

def normalize (d, n = 1):
  total = n * float(sum(d.values()))
  d = dict(map(lambda x: (x[0],x[1]/total), d.items()))
  return d

def convertMatList(mat): #assumes first col is aligned
  edges = []
  for y in range(len(mat)):
    if y < len(mat)-1 and 0 < len(mat[y+1]):
      edges.append(('y', (mat[y][0][0], mat[y+1][0][0])))
    for x in range(len(mat[y])):
      edges.append(('x', mat[y][x]))
      if y < len(mat)-1 and x < len(mat[y+1]):
        edges.append(('y', (mat[y][x][1], mat[y+1][x][1])))
  return edges

def convertMatPos(mat): #assumes first col is aligned
  pos = {}
  for y in range(len(mat)):
    for x in range(len(mat[y])):
      pos[mat[y][x][0]] = (y,x)
    pos[mat[y][len(mat[y])-1][1]] = (y,len(mat[y]))
  return pos

def plotResults(c, p, g, y, costEval):
  plt.figure(1)
  plt.subplot(211)
  plt.plot(y,c, 'r', y,p, 'g', y, g, 'b')
  plt.annotate("Kruskal", (y[-2],c[-2]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
  plt.annotate("Prim", (y[-3],p[-3]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
  plt.annotate("Greedy1D", (y[-4],g[-4]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
  plt.xlabel("sqrt(no. strips)")
  plt.ylabel("Percent correct edges")

  #plt.subplot(312)
  #plt.plot(y,ic, 'r', y,ip,'g', y, ig, 'b')
  #plt.annotate("Correct", (y[-2],ic[-2]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
  #plt.annotate("Prim", (y[-3],ip[-3]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
  #plt.annotate("Greedy1D", (y[-4],ig[-4]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
  #plt.xlabel("sqrt(no. strips)")
  #plt.ylabel("Internal Cost")

  plt.subplot(212)
  plt.plot( y, costEval, 'r')
  plt.annotate("CostEval", (y[4],costEval[4]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

  plt.show()


if __name__ == "__main__":
  arg = ""
  if len(sys.argv) == 1:
    arg = "1"
    print 'No argument given, assuming "1"'
  else:
    arg= sys.argv[1]
  if "1" in arg:
    reps = 100
    if len(sys.argv) < 2:
      print "Number of repetitions not specified, assuming 100"
    elif not sys.argv[2].isdigit() or int(sys.argv[2].isdigit()) < 1:
      raise Exception("Given number of repetitions: " + sys.argv[2] + " is an invalid number.")
    else:
      reps = int(sys.argv[2])    
    correct = 0
    g1D = 0
    prm = 0
    for i in range(reps):
      print i
      page = pages.GaussianPage(2,5,5, True) # turning noise off causes very odd behaviour, should investigate
      g1DMat = search.picker("g1D", page)
      page.heapify()
      (pos, edges) = search.picker("prim", page)
      #print g1DMat
      #print edges
      correct += sum(page.totalCost)
      g1D += sum(page.calcCostMat(g1DMat))
      prm += sum(page.calcCostList(edges))
    print correct/reps, prm/reps, g1D/reps
  elif "2" in arg:
    sx = 20
    sy = 2
    page = pages.ImagePage(sx, sy, "SampleDocs/text3.jpg")
    
    #print randCostX.items()[:3]
    #print randCostY.items()[:3]
    #bitCostX, bitCostY = cost.picker("bit", page) 
    #gausCostX, gausCostY = cost.picker("gaus", page) 
    #blackBitCostX, blackBitCostY = cost.picker("blackBit", page) 
    #blackGausCostX, blackGausCostY = cost.picker("blackGaus", page) 
    #print costX, costY, "\n"

    print "RandCost: ",
    page.setCost("rand")
    cost.evaluateCost(page, sx, sy)

    print "ProcRandCost: ",
    page.setCost("rand", True)
    cost.evaluateCost(page, sx, sy)

    print "BitCost: ",
    page.setCost("bit")    
    cost.evaluateCost(page, sx, sy)

    print "ProcBitCost: ",
    page.setCost("bit", True)
    #print sorted(page.costX.items(), key=lambda x:x[1])[:100]
    cost.evaluateCost(page, sx, sy)

    print "GausCost: ",
    page.setCost("gaus")
    cost.evaluateCost(page, sx, sy)

    print "ProcGausCost: ",
    page.setCost("gaus", True)
    cost.evaluateCost(page, sx, sy)

    print "BlackBitCost: ",
    page.setCost("blackBit")
    cost.evaluateCost(page, sx, sy)

    print "ProcBlackBitCost: ",
    page.setCost("blackBit", True)
    cost.evaluateCost(page, sx, sy)

    print "BlackGausCost: ",
    page.setCost("blackGaus")
    cost.evaluateCost(page, sx, sy)

    print "ProcBlackGausCost: ",
    page.setCost("blackGaus", True)
    cost.evaluateCost(page, sx, sy)

    print "BlackRowCost: ",
    page.setCost("blackRow")
    cost.evaluateCost(page, sx, sy)

    print "ProcBlackRowCost: ",
    page.setCost("blackRow", True)
    cost.evaluateCost(page, sx, sy)

    print "BlackRowGausCost: ",
    page.setCost("blackRowGaus")
    cost.evaluateCost(page, sx, sy)

    print "ProcBlackRowGausCost: ",
    page.setCost("blackRowGaus", True)
    cost.evaluateCost(page, sx, sy)

    #print sorted(gausCostX.items()),"\n", sorted(bitCostX.items()), "\n"
  elif "3" in arg:
    c = []
    p = []
    g = []
    ec = []
    ep = []
    eg = []
    ek = []
    ic = []
    ip = []
    ig = []
    k = []
    ik = []
    y = []
    costEval = []
    #prior, prx, pry = getPrediction([Image.open("SampleDocs/text31.jpg").convert("1")])

    for i in range(2,16):
      print i
      sx = i
      sy = i
      correct = 0
      g1D = 0
      prm = 0
      reps = 1
      intCorrect = 0
      intG1D = 0
      intPrm = 0
      
      page = pages.ImagePage(sx, sy, "SampleDocs/text31.jpg")
      #page.setCost("prediction", prx, pry, prior)
      page.setCost("blackGaus")
      #page.output(page.states[(1,1)])
      #print cost.indivPicker(page.costType,(1,1), (-42,-42), "x",  page, False)
      #g1DMat = search.picker("g1D", page)
      #page.heapify()
      (pos, edges) = search.picker("kruskal", page)
      #page.heapify()
      #(kpos, kedges) = search.picker("kruskal", page)
      #print g1DMat
      #print edges
      #print pos
      #print convertMatPos(g1DMat)
      correct += sum(page.totalCost)
      #g1D += sum(page.calcCostMat(g1DMat))
      prm += sum(page.calcCostList(pos))
      intCorrect += sum(page.totalInternalCost)
      #intG1D += sum(page.calcCostMat(g1DMat, True))

      intPrm += sum(page.calcCostList(pos, True))
      costEval.append(cost.evaluateCost(page, sx, sy)[0])
      print intCorrect/reps, intPrm/reps#, intG1D/reps
      print correct/reps, prm/reps#, g1D/reps
      print "cost eval", cost.evaluateCost(page, sx, sy)
      edgeP = page.calcCorrectEdges(pos)
      #edgeK = page.calcCorrectEdges(kpos)
      #edgeG = page.calcCorrectEdges(convertMatPos(g1DMat))
      edgeC = 1.0
      print "edges", edgeP#, edgeK, edgeG
      #print sorted(page.calcCostList(convertMatPos(g1DMat), False)) == sorted(page.calcCostMat(g1DMat, False))
      c.append(correct)
      p.append(prm)
      #g.append(g1D)
      ec.append(edgeC)
      ep.append(edgeP)
      #eg.append(edgeG)
      #ek.append(edgeK)
      ic.append(intCorrect)
      ip.append(intPrm)
      #ig.append(intG1D)
      y.append(i*i)
      #print sum(page.calcCostList({(0, 1): (0, 1), (1, 0): (1, 0), (0, 0): (0, 0), (1, 1): (1, 1)}))

    #print ' '.join(map(str,c))
    #print ' '.join(map(str,ic))
    #print ' '.join(map(str,p))
    #print ' '.join(map(str,ip))
    #print ' '.join(map(str,ep))


    plt.plot(y,c, 'r', y,p, 'g', y, ic, 'b', y, ip, 'm')
    plt.annotate("Total Correct cost", (y[3],c[3]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Total Kruskal cost", (y[6],p[6]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Interior Correct cost", (y[9],ic[9]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Interior Kruskal cost", (y[13],ip[13]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")
    plt.ylabel("Cost")
    plt.show()

  elif "4" in arg:
    ys = []
    #prior, prxl, pryl, prxr, pryr = getPrediction([Image.open("SampleDocs/text31.jpg").convert("1")])
    for i in range(2,21):
      sx = i
      sy = i

      page = pages.ImagePage(sx, sy, "SampleDocs/text31.jpg")
      #print page.whites

      #print verifCost.checkAll(page)
      page.setCost("gaus")
      #page.setCost("blackGaus")
      #page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
      
      #g1DMat = search.picker("g1D", page)
      #page.heapify()
      #(pos, edges) = search.picker("kruskal", page)
      #page.heapify()
      #(kPos, kEdges) = search.picker("kruskal", page, True)
      #print edges
      #print sorted(pos.items(), key=lambda x: x[1])
      #print g1DMat
      #print sum(page.totalInternalCost), sum(page.calcCostList(kPos, True)), sum(page.calcCostList(pos, True)), sum(page.calcCostMat(g1DMat, True))
      #print sum(page.totalCost),sum(page.calcCostList(kPos)), sum(page.calcCostList(pos)), sum(page.calcCostMat(g1DMat))
      cc = cost.evaluateCost(page, sx, sy)
      print i, cc#,page.calcCorrectEdges(kPos, True), page.calcCorrectEdges(pos), page.calcCorrectEdges(convertMatPos(g1DMat))
      ys.append(cc)
      #page.vizPos(kPos, "kruskal")
      #page.vizPos(pos, "prim")
      #page.vizPos(convertMatPos(g1DMat), "g1d")

      #print page.extractHeap([((0,1),(0,1)),((0,0),(2,0))])
      #print page.verifyData()

      #plot = page.costX.values()
      #mn = min(plot)
      #plot = map(lambda x: float(x)/mn, plot)

      #plot = page.pieceDistX((1,4)).values()
      #print plot

      #print page.pieceDistX((0,0))
      #sCosts = cost.processCostX(page)
      
      #n, bins, patches = plt.hist(plot, 2000, facecolor='g', alpha=1)
      #plt.xscale('log')
      #plt.yscale('log')
      #plt.xlabel('')
      #plt.ylabel('')
      #plt.grid(True)
      #plt.show()
    print ys
  elif "5" in arg:
    #px, py = getPercents([Image.open("SampleDocs/text3.jpg").convert("1")])
    #print "orig px", '\n'.join(map(str,sorted(px.items(), key=lambda x : x[1]))), '\n======'
    #print "orig py",  '\n'.join(map(str,sorted(py.items(), key=lambda x : x[1]))), '\n======'
    #prior, prxl, pryl, prxr, pryr = getPrediction([Image.open("SampleDocs/text31.jpg").convert("1")])
    #print prior
    #print "orig prxr", '\n'.join(map(str,sorted(prxr.items(), key=lambda x : x[1][0.0]))), '\n======'
    #print "orig prxl",  '\n'.join(map(str,sorted(prxl.items(), key=lambda x : x[1][0.0]))), '\n======'
    ns = []
    xs = []
    ys = []
    for i in range(16,21):
      start = time.clock()
      sx = i*i
      sy = 1
      #print "ff"
      page = pages.ImagePage(sx, sy, "SampleDocs/text31.jpg")
      #rr = page.confEdges(4)
      #print i, rr
      #xs.append(rr)
      #npx, npy = getPercents(page.states.values())
      #print "piece px", sorted(px, key=lambda x : x[1]), '\n======'
      #print "piece py", sorted(py, key=lambda x : x[1]), '\n======'
      #print "qq"
      #page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
      page.setCost("blackGaus")
      (pPos, pEdges) = search.picker("prim", page)
      #page.heapify()
      #(kPos, kEdges) = search.picker("kruskal", page)
      corrp = page.calcCorrectEdges(pPos)
      corrk = 0#page.calcCorrectEdges(kPos)
      ns.append(i*i)
      xs.append(corrp)
      ys.append(corrk)
      sys.stderr.write(str(i) + " " + str(corrp) + " " + str(corrk) + " " + str(time.clock() - start) + "\n")

    print "prim", xs
    print "kruskal", ys
    plt.plot(ns,xs, 'r-')#, ns,ys, 'g-')
    
    plt.xlabel("Number of shreds")
    plt.ylabel("Proportion of shreds with 4 or less black pixels on each edge")
    plt.show()
 
  elif "6" in arg:
    #ex1, ey1 = genEdges(5,5)
    #for i in range(1):
      #ex2, ey2 = simKruskal(0.9,20,20)
      #print ex1, ey1
      #print ex2, ey2
      #print simError(ex1,ey1,ex2,ey2)
    #assert False
    sx = 25
    sy = 1
    count = 50
    xs = []
    ys = []
    ts = []
    ns = []

    ex1, ey1 = genEdges(sx,sy)
    for f in np.arange(0.0, 1.001, 0.01):
      sumXp = 0.0
      sumYp = 0.0
      sumTp = 0.0
      sumXk = 0.0
      sumYk = 0.0
      sumTk = 0.0
      for i in range(count):
        ex2, ey2 = simKruskal(f,sx,sy)
        ex3, ey3 = simPrim(f,sx,sy)
        #print ex1, ey1
        #print ex2, ey2
        xk,yk,tk = simError(ex1,ey1,ex2,ey2)
        sumXk += xk
        sumYk += yk
        sumTk += tk
        xp,yp,tp = simError(ex1,ey1,ex3,ey3)
        sumXp += xp
        sumYp += yp
        sumTp += tp

      sumXp /= count
      sumYp /= count
      sumTp /= count

      sumXk /= count
      sumYk /= count
      sumTk /= count

      print "Prim", f, sumXp, sumYp, sumTp
      print "Kruskal", f, sumXk, sumYk, sumTk
      xs.append(sumTp)
      ys.append(sumTk)
      #ts.append(sumT)
      ns.append(f)

    print xs
    print ys
    plt.figure(1)
    plt.plot(ns,xs, 'r-', ns,ys, 'g-', ns, [1-x for x in ns], 'b-')
    plt.show()
  elif "7" in arg:
    a = [1.0, 0.95924999999999994, 0.86175000000000024, 0.84450000000000003, 0.78900000000000015, 0.71624999999999983, 0.71700000000000019, 0.70824999999999994, 0.64249999999999996, 0.63174999999999992, 0.55525000000000002, 0.56450000000000033, 0.55025000000000002, 0.49775000000000008, 0.51724999999999999, 0.47599999999999992, 0.48950000000000016, 0.45925000000000005, 0.42825000000000008, 0.44024999999999997, 0.39824999999999994, 0.39075000000000015, 0.38525000000000004, 0.35549999999999998, 0.37799999999999989, 0.33949999999999997, 0.34899999999999998, 0.30175000000000018, 0.3040000000000001, 0.30774999999999997, 0.29999999999999993, 0.28549999999999992, 0.29249999999999993, 0.25200000000000006, 0.30374999999999991, 0.25999999999999995, 0.24599999999999997, 0.24399999999999999, 0.23824999999999999, 0.2382500000000001, 0.24549999999999994, 0.21424999999999997, 0.21799999999999994, 0.20200000000000004, 0.19724999999999987, 0.19624999999999995, 0.18700000000000003, 0.18899999999999989, 0.18075000000000002, 0.17675000000000002, 0.17050000000000004, 0.17450000000000002, 0.1717499999999999, 0.16449999999999998, 0.15525000000000003, 0.15024999999999999, 0.14849999999999997, 0.15274999999999997, 0.14024999999999996, 0.14074999999999999, 0.13100000000000001, 0.1235, 0.12924999999999998, 0.11574999999999992, 0.11724999999999999, 0.10725000000000001, 0.098500000000000018, 0.11399999999999995, 0.09425, 0.10250000000000002, 0.092250000000000013, 0.095499999999999988, 0.086749999999999994, 0.089250000000000065, 0.085750000000000007, 0.076249999999999998, 0.070500000000000007, 0.067999999999999991, 0.063999999999999974, 0.062249999999999986, 0.056250000000000008, 0.056250000000000001, 0.056749999999999988, 0.052999999999999999, 0.044499999999999984, 0.04474999999999997, 0.0395, 0.032249999999999973, 0.037749999999999964, 0.031999999999999973, 0.027249999999999983, 0.027249999999999969, 0.023499999999999986, 0.019249999999999993, 0.015749999999999997, 0.015499999999999998, 0.008750000000000006, 0.01125, 0.0080000000000000036, 0.0032500000000000003, 0.0]
    b = [1.0, 0.95650000000000024, 0.92099999999999993, 0.90675000000000039, 0.87725000000000009, 0.81824999999999992, 0.79049999999999965, 0.77375000000000016, 0.74724999999999975, 0.75525000000000009, 0.71325000000000049, 0.70825000000000016, 0.66849999999999998, 0.65724999999999978, 0.6382500000000001, 0.65750000000000031, 0.61624999999999985, 0.6472500000000001, 0.59250000000000003, 0.59449999999999992, 0.56799999999999995, 0.57774999999999999, 0.56950000000000001, 0.56149999999999989, 0.53625000000000012, 0.51849999999999996, 0.49825000000000008, 0.50075000000000014, 0.47950000000000009, 0.48399999999999999, 0.47225000000000017, 0.45674999999999988, 0.44550000000000012, 0.4517500000000001, 0.44950000000000018, 0.43250000000000016, 0.43224999999999997, 0.39850000000000002, 0.40100000000000008, 0.39624999999999999, 0.39775000000000021, 0.38274999999999987, 0.38425000000000009, 0.36475000000000007, 0.36575000000000002, 0.34900000000000003, 0.34449999999999981, 0.32899999999999996, 0.32450000000000012, 0.31300000000000006, 0.31049999999999994, 0.30600000000000011, 0.30399999999999999, 0.27724999999999977, 0.27250000000000008, 0.27175000000000005, 0.26949999999999991, 0.26525000000000004, 0.25974999999999993, 0.25724999999999992, 0.23725000000000004, 0.23775000000000004, 0.22725000000000012, 0.23549999999999996, 0.2205, 0.21399999999999988, 0.21475000000000005, 0.18275000000000002, 0.19450000000000001, 0.185, 0.18375000000000019, 0.16824999999999996, 0.16575000000000004, 0.1515, 0.16674999999999987, 0.15475, 0.13674999999999998, 0.13599999999999998, 0.13424999999999998, 0.13125000000000001, 0.12024999999999995, 0.11225000000000003, 0.099999999999999978, 0.096000000000000002, 0.090750000000000011, 0.089749999999999941, 0.080749999999999975, 0.087499999999999967, 0.07400000000000001, 0.068500000000000019, 0.058749999999999976, 0.058499999999999955, 0.051249999999999983, 0.038999999999999972, 0.036499999999999984, 0.031749999999999966, 0.02424999999999998, 0.017499999999999988, 0.012000000000000002, 0.0075000000000000023, 0.0]
    c = [1.0, 0.97250000000000003, 0.94999999999999996, 0.88749999999999984, 0.87333333333333329, 0.87083333333333313, 0.83999999999999997, 0.80666666666666642, 0.78833333333333333, 0.75833333333333297, 0.73833333333333317, 0.72833333333333328, 0.69583333333333341, 0.66749999999999998, 0.64583333333333326, 0.63333333333333353, 0.64166666666666661, 0.62916666666666676, 0.58583333333333332, 0.57833333333333325, 0.60999999999999988, 0.59083333333333343, 0.57999999999999996, 0.56583333333333319, 0.51249999999999996, 0.50583333333333325, 0.54749999999999999, 0.47833333333333333, 0.51083333333333336, 0.47333333333333322, 0.47166666666666662, 0.45666666666666667, 0.41166666666666657, 0.42416666666666658, 0.45250000000000007, 0.41499999999999998, 0.42083333333333328, 0.38249999999999995, 0.41749999999999998, 0.40250000000000002, 0.36749999999999994, 0.36333333333333345, 0.35249999999999998, 0.36499999999999999, 0.35666666666666669, 0.32166666666666666, 0.33749999999999991, 0.33833333333333343, 0.31833333333333336, 0.32000000000000001, 0.30583333333333323, 0.30250000000000005, 0.25333333333333335, 0.2583333333333333, 0.25083333333333346, 0.255, 0.28083333333333338, 0.26083333333333336, 0.25083333333333335, 0.22333333333333333, 0.21083333333333332, 0.20999999999999996, 0.20499999999999999, 0.19833333333333342, 0.18583333333333341, 0.17666666666666664, 0.19916666666666658, 0.1783333333333334, 0.1883333333333333, 0.16250000000000003, 0.18416666666666667, 0.15083333333333335, 0.16, 0.16333333333333333, 0.13750000000000001, 0.13250000000000001, 0.12333333333333338, 0.11166666666666668, 0.10249999999999999, 0.10999999999999996, 0.10583333333333332, 0.088333333333333361, 0.098333333333333356, 0.090833333333333321, 0.07333333333333332, 0.072499999999999995, 0.07333333333333332, 0.066666666666666666, 0.055833333333333332, 0.058333333333333341, 0.047500000000000001, 0.043333333333333328, 0.054166666666666634, 0.04250000000000001, 0.026666666666666665, 0.022499999999999999, 0.018333333333333333, 0.013333333333333331, 0.011666666666666667, 0.0033333333333333331, 0.00083333333333333328]
    d = [1.0, 0.96833333333333338, 0.97416666666666663, 0.93916666666666659, 0.93999999999999984, 0.89333333333333331, 0.89083333333333325, 0.86416666666666653, 0.87333333333333341, 0.84916666666666674, 0.85333333333333317, 0.83750000000000002, 0.82833333333333348, 0.78333333333333333, 0.78750000000000009, 0.76500000000000001, 0.76333333333333342, 0.74666666666666659, 0.75083333333333324, 0.72333333333333338, 0.71499999999999997, 0.66583333333333339, 0.6958333333333333, 0.66833333333333345, 0.66083333333333327, 0.65666666666666662, 0.65916666666666668, 0.62500000000000011, 0.61916666666666653, 0.62666666666666671, 0.61249999999999993, 0.64333333333333331, 0.60500000000000009, 0.58916666666666684, 0.56000000000000005, 0.56833333333333336, 0.53083333333333338, 0.5558333333333334, 0.54416666666666658, 0.52499999999999991, 0.53750000000000009, 0.51249999999999996, 0.52916666666666656, 0.47583333333333344, 0.48583333333333334, 0.45750000000000002, 0.44166666666666665, 0.47499999999999987, 0.44500000000000001, 0.43166666666666681, 0.44583333333333319, 0.41166666666666663, 0.38333333333333336, 0.40416666666666656, 0.38916666666666672, 0.37583333333333335, 0.35499999999999998, 0.37083333333333329, 0.35833333333333345, 0.3600000000000001, 0.33916666666666673, 0.34083333333333321, 0.32583333333333336, 0.31083333333333324, 0.31000000000000005, 0.3075, 0.27750000000000002, 0.28083333333333338, 0.27083333333333331, 0.23999999999999994, 0.25416666666666665, 0.2558333333333333, 0.24083333333333329, 0.22583333333333333, 0.21416666666666664, 0.21583333333333329, 0.20499999999999999, 0.19000000000000003, 0.18250000000000002, 0.18166666666666667, 0.19166666666666665, 0.16333333333333336, 0.16833333333333333, 0.13749999999999998, 0.12000000000000004, 0.13999999999999999, 0.097500000000000003, 0.10416666666666666, 0.10166666666666666, 0.099166666666666681, 0.095000000000000001, 0.071666666666666642, 0.078333333333333352, 0.055833333333333339, 0.059166666666666673, 0.036666666666666681, 0.03500000000000001, 0.031666666666666683, 0.014166666666666662, 0.0083333333333333332, 0.0]
    ns = np.arange(0.0, 1.001, 0.01)
    rns = [1-x for x in ns]
    
    pbgcc = [1.0, 0.416666666667, 1.0, 0.825, 0.966666666667, 0.571428571429, 0.660714285714, 0.611111111111, 0.494444444444, 0.359090909091, 0.371212121212, 0.304487179487, 0.266483516484, 0.316666666667, 0.322916666667, 0.257352941176, 0.218954248366, 0.191520467836, 0.192105263158]
    ppcc = [1.0, 0.666666666667, 0.916666666667, 1.0, 0.783333333333, 0.738095238095, 0.633928571429, 0.555555555556, 0.672222222222, 0.445454545455, 0.5, 0.38141025641, 0.423076923077, 0.430952380952, 0.63125, 0.4375, 0.34477124183, 0.210526315789, 0.284210526316 ]
    pps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.98989898989899, 0.9916666666666667, 0.993006993006993, 0.9880952380952381, 0.9538461538461539, 0.9910714285714286, 0.9882352941176471, 0.9895833333333334, 0.9845201238390093, 0.9777777777777777, 0.9799498746867168 ]
    ln = len(pps)
    xs = [i*i for i in range(2, ln + 2)]
    plt.plot(xs,pbgcc, 'g-*', xs,ppcc, 'r-H', xs, pps, 'b-d')#, ns, c, 'm-', ns, d, 'c-')
    plt.annotate("Cross-cut - BestGausCost", (xs[12],pbgcc[12]), xytext = (70, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Cross-cut - ProbScore", (xs[13],ppcc[13]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Strip - BestGausCost", (xs[14], pps[14]), xytext = (70, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")
    plt.ylabel("% correct edges")
    
    """
    p = [x[0] for x in [(0.8, 0.0), (0.7749999999999999, 0.0), (0.7598039215686274, 41.270833333333336) , (0.975, 2.45), (0.7711711711711711, 19.470833333333335), (0.6461904761904763, 54.11904761904762), (0.6092032967032969, 92.56473214285714), (0.5316734417344179, 45.14930555555556), (0.6681518151815183, 55.84305555555556), (0.6105067064083457, 55.29090909090909), (0.5205067920585166, 73.20359848484848), (0.4569570135746596, 61.10657051282051), (0.44915490600769725, 57.10302197802198), (0.4560577328276446, 69.84940476190476), (0.5246473735408566, 64.28333333333333), (0.38427991886409757, 77.859375), (0.398109602815484, 60.91053921568628), (0.2955881877806843, 87.99305555555556), (0.3830013781336137, 75.01381578947368)] ]
    q = [x[0] for x in [(0.75, 1.8038133282386566), (0.6375000000000001, 3.3425124410695517), (0.6458333333333333, 3.7734125617064875), (0.825, 3.850164457453277), (0.7438034188034188, 1.3747400868759352), (0.6380772005772007, 3.077451618253361), (0.6554950105042018, 1.8730934144411424), (0.5480492365019806, 2.622829437435161), (0.664290577342048, 2.8995891642396208), (0.5844615384615385, 4.1993985788788875), (0.5716104258178918, 2.581604924076438), (0.47467194438519866, 2.631028637773309), (0.5029462800682958, 2.6249431552190754), (0.49803137443307877, 2.071597279028274), (0.5990971453033893, 1.610507205822009), (0.4543868542575038, 2.234360306630231), (0.447287362649943, 1.7974495531844519), (0.33342581207942107, 2.949557047232156), (0.41095526460619297, 3.9095273918431)] ]
    r = [x[0] for x in [(0.75, 1.5), (0.6375000000000001, 5.583333333333333), (0.6458333333333333, 5.625), (0.6799999999999999, 5.075), (0.710989010989011, 2.1), (0.606677713820571, 2.988095238095238), (0.5789996700434561, 2.1607142857142856), (0.49700418850571304, 2.8819444444444446), (0.5779050567595461, 3.0944444444444446), (0.5208831353831352, 4.213636363636364), (0.5099144917079702, 3.0), (0.43490715884544184, 2.8076923076923075), (0.43365558242670266, 2.697802197802198), (0.4134172319871909,2.461904761904762), (0.5225286406387423, 2.0458333333333334), (0.3619443696207882, 2.4981617647058822), (0.37367085904016933, 2.076797385620915), (0.27045133427066786, 2.953216374269006), (0.32327951086021617, 4.102631578947369)] ]
    
    xs = [i*i for i in range(2, 21)]

    plt.plot(xs,p, 'g-*', xs,q, 'r-H', xs, r, 'b-d')#, ns, c, 'm-', ns, d, 'c-')
    plt.annotate("BlackGausCost", (xs[13],p[13]), xytext = (45, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("ProbScore", (xs[14],q[14]), xytext = (45, 7), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("GausCost", (xs[7], r[7]), xytext = (22, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")
    plt.ylabel("% correct edges")
    plt.yticks([0.25,0.35,0.45,0.55,0.65])
    """
    """
    plt.plot(ns,a, 'g-', ns,b, 'r-', ns, rns, 'b-')#, ns, c, 'm-', ns, d, 'c-')
    ma = [a[i] for i in range(len(a)) if i % 5 == 0]
    mb = [b[i] for i in range(len(a)) if i % 5 == 0]
    mrns = [rns[i] for i in range(len(a)) if i % 5 == 0]
    mns = [ns[i] for i in range(len(a)) if i % 5 == 0]
    plt.plot(mns,ma, 'g*', mns,mb, 'rH', mns, mrns, 'bd')#, ns, c, 'm-', ns, d, 'c-')
    plt.annotate("Prim", (ns[20],a[20]), xytext = (0, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Kruskal", (ns[60],b[60]), xytext = (40, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("No cascading", (ns[30],rns[30]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Prim 1*25", (ns[60],c[60]), xytext = (-30, -10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Kruskal 1*25", (ns[50],d[50]), xytext = (30, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Cost/score function error rate")
    plt.ylabel("% correct edges")
    """
    a = plt.gca()
    a.set_ylim([0.0,1.05])
    plt.show()
  else:
    print 'Unrecognized argument'
