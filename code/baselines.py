import sys
from matplotlib import pyplot as plt
import search
import cost
import verifCost
import pages
import time
from PIL import Image

def getPercents(im):
  perX = {}
  perY = {}
  threeX = {}
  threeY = {}
  w,h = im.size
  ld = list(im.getdata())
  print w,h, len(ld)
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
        perX[n] = 1
      try:
        threeX[t1] += 1
      except:
        threeX[t1] = 1
      try:
        threeX[t2] += 1
      except:
        threeX[t2] = 1

  for r in range(h-1):
    for c in range(1,w-1):
      n = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div,bd[r+1][c-1]/div,bd[r+1][c]/div,bd[r+1][c+1]/div)
      t1 = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div)
      t2 = (bd[r+1][c-1]/div,bd[r+1][c]/div,bd[r+1][c+1]/div)
      try:
        perY[n] += 1
      except:
        perY[n] = 1
      try:
        threeY[t1] += 1
      except:
        threeY[t1] = 1
      try:
        threeY[t2] += 1
      except:
        threeY[t2] = 1

  perX = normalize(perX)
  perY = normalize(perY)
  threeX = normalize(threeX)
  threeY = normalize(threeY)

  surX = {}
  surY = {}
  for n in perX:
    t1 = (n[0],n[1],n[2])
    t2 = (n[3],n[4],n[5])
    surX[n] = (threeX[t1] * threeX[t2]) / perX[n]

  for n in perY:
    t1 = (n[0],n[1],n[2])
    t2 = (n[3],n[4],n[5])
    surY[n] = (threeY[t1] * threeY[t2]) / perY[n]

  #print "sX"
  #print sorted(surX.items(), key=lambda x:x[1])
  #print "sY"
  #print sorted(surY.items(), key=lambda x:x[1])
  return surX, surY

def normalize (d):
  total = float(sum(d.values()))
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
    for i in range(2,33):
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
      
      page = pages.ImagePage(sx, sy, "SampleDocs/text3.jpg")
      page.setCost("blackGaus")
      #page.output(page.states[(1,1)])
      #print cost.indivPicker(page.costType,(1,1), (-42,-42), "x",  page, False)
      g1DMat = search.picker("g1D", page)
      page.heapify()
      (pos, edges) = search.picker("prim", page)
      page.heapify()
      (kpos, kedges) = search.picker("kruskal", page)
      #print g1DMat
      #print edges
      #print pos
      #print convertMatPos(g1DMat)
      correct += sum(page.totalCost)
      g1D += sum(page.calcCostMat(g1DMat))
      prm += sum(page.calcCostList(pos))
      intCorrect += sum(page.totalInternalCost)
      intG1D += sum(page.calcCostMat(g1DMat, True))
            
      intPrm += sum(page.calcCostList(pos, True))
      costEval.append(cost.evaluateCost(page, sx, sy)[0])
      print intCorrect/reps, intPrm/reps, intG1D/reps
      print correct/reps, prm/reps, g1D/reps
      print "cost eval", cost.evaluateCost(page, sx, sy)
      edgeP = page.calcCorrectEdges(pos)
      edgeK = page.calcCorrectEdges(kpos)
      edgeG = page.calcCorrectEdges(convertMatPos(g1DMat))
      edgeC = 1.0
      print "edges", edgeK, edgeP, edgeG
      #print sorted(page.calcCostList(convertMatPos(g1DMat), False)) == sorted(page.calcCostMat(g1DMat, False))
      c.append(correct)
      p.append(prm)
      g.append(g1D)
      ec.append(edgeC)
      ep.append(edgeP)
      eg.append(edgeG)
      ek.append(edgeK)
      ic.append(intCorrect)
      ip.append(intPrm)
      ig.append(intG1D)
      y.append(i)
      #print sum(page.calcCostList({(0, 1): (0, 1), (1, 0): (1, 0), (0, 0): (0, 0), (1, 1): (1, 1)}))

    print ec
    print ep
    print eg

    #plt.plot(y,ec, 'r', y,ep, 'g', y, eg, 'b')
    #plt.show()
    plotResults(ek, ep, eg, y, costEval)

  elif "4" in arg:
    sx = 10
    sy = 10

    page = pages.ImagePage(sx, sy, "SampleDocs/p01.png")
    print page.whites

    #print verifCost.checkAll(page)
    page.setCost("blackGaus")

    g1DMat = search.picker("g1D", page)
    page.heapify()
    (pos, edges) = search.picker("prim", page)
    page.heapify()
    (kPos, kEdges) = search.picker("kruskal", page, True)
    #print edges
    #print sorted(pos.items(), key=lambda x: x[1])
    #print g1DMat
    print sum(page.totalInternalCost), sum(page.calcCostList(kPos, True)), sum(page.calcCostList(pos, True)), sum(page.calcCostMat(g1DMat, True))
    print sum(page.totalCost),sum(page.calcCostList(kPos)), sum(page.calcCostList(pos)), sum(page.calcCostMat(g1DMat))
    print cost.evaluateCost(page, sx, sy),page.calcCorrectEdges(kPos, True), page.calcCorrectEdges(pos), page.calcCorrectEdges(convertMatPos(g1DMat))

    page.vizPos(kPos, "kruskal")
    page.vizPos(pos, "prim")
    page.vizPos(convertMatPos(g1DMat), "g1d")

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
  elif "5" in arg:
    px, py = getPercents(Image.open("SampleDocs/text3.jpg").convert("1"))
    for i in range(2,21):
      start = time.clock()
      sx = i
      sy = i
      page = pages.ImagePage(sx, sy, "SampleDocs/text3.jpg")
      #page.setCost("percent", False, px, py)
      page.setCost("blackGaus")
      (kpos, kedges) = search.picker("kruskal", page)

      sys.stderr.write(str(i) + " " + str(page.calcCorrectEdges(kpos)) + " " + str(time.clock() - start) + "\n")
      
  else:
    print 'Unrecognized argument'
