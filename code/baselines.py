import sys
from matplotlib import pyplot as plt

import search
import cost
import pages

if __name__ == "__main__":
  arg = ""
  if len(sys.argv) == 1:
    arg = "g"
    print 'No argument given, assuming "g"'
  else:
    arg= sys.argv[1]
  if "g" in arg:
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
  elif "p" in arg:
    sx = 1
    sy = 10
    page = pages.ImagePage(sx, sy, "text3.jpg")
    
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

    print "BitCost: ",
    page.setCost("bit")    
    cost.evaluateCost(page, sx, sy)

    print "GausCost: ",
    page.setCost("gaus")
    cost.evaluateCost(page, sx, sy)

    print "BlackBitCost: ",
    page.setCost("blackBit")
    cost.evaluateCost(page, sx, sy)

    print "BlackGausCost: ",
    page.setCost("black")
    cost.evaluateCost(page, sx, sy)
    #print sorted(gausCostX.items()),"\n", sorted(bitCostX.items()), "\n"
  elif "s" in arg:
    c = []
    p = []
    g = []
    ic = []
    ip = []
    ig = []
    y = []
    for i in range(2,15):
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

      page = pages.ImagePage(sx, sy, "text2.jpg")
      page.setCost("blackRow")
      #page.output(page.states[(1,1)])
      #print cost.indivPicker(page.costType, page.states[(1,1)], page.blank, "x", False, page.blank, page.sizeX, page.sizeY)
      g1DMat = search.picker("g1D", page)
      page.heapify()
      (pos, edges) = search.picker("prim", page)
      print g1DMat
      print edges
      #print pos
      correct += sum(page.totalCost)
      g1D += sum(page.calcCostMat(g1DMat))
      prm += sum(page.calcCostList(pos))
      intCorrect += sum(page.totalInternalCost)
      intG1D += sum(page.calcCostMat(g1DMat, True))
      intPrm += sum(page.calcCostList(pos, True))
      print intCorrect/reps, intPrm/reps, intG1D/reps
      print correct/reps, prm/reps, g1D/reps
      c.append(correct)
      p.append(prm)
      g.append(g1D)
      ic.append(intCorrect)
      ip.append(intPrm)
      ig.append(intG1D)
      y.append(i)
      #print sum(page.calcCostList({(0, 1): (0, 1), (1, 0): (1, 0), (0, 0): (0, 0), (1, 1): (1, 1)}))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(y,c, 'r', y,p, 'g', y, g, 'b')

    plt.subplot(212)
    plt.plot(y,ic, 'r', y,ip,'g', y, ig, 'b')

    plt.show()
  else:
    print 'Unrecognized argument, use "g" for gaussian simulation or "p" image processing'
