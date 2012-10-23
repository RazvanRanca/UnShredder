from PIL import Image
import matplotlib.pyplot as plt
import random as r
import heapq as hq
import cost

class ImagePage():
  def __init__(self,sx, sy, filename, cType = None):
    img = Image.open(filename).convert("1")
    pieces = {}
    w,h = img.size
    #print w, h
    h = h / sy
    w = w / sx
    blank = Image.new("1", (w,h), 255)
    for i in range(sy):
      for j in range(sx):
        g = img.crop((j*w,i*h,(j+1)*w,(i+1)*h))
        pieces[(i,j)] = g
    #print pieces
    #output(pieces[(0,1)])
    blanks = []
    #for i in range(0,sx):
    #  pieces[(-1,i)] = blank
    #  pieces[(sy,i)] = blank
    #  blanks += [(-1,i),(sy,i)]
    #for i in range(0,sy):
    #  pieces[(i,-1)] = blank
    #  pieces[(i,sx)] = blank
    #  blanks += [(i,-1),(i,sx)]

    self.blankPos = blanks
    self.corners = [(-1,-1),(sy,-1),(-1,sx),(sy,sx)]
    self.sizeX = sx
    self.sizeY = sy
    self.states = pieces
    self.orig = img
    self.blank = blank
    if cType != None:
      self.setCost(cType)
    self.getBlanks()

  def getAllStates(self): # returns a list of all (i,j) positions considered, including blanks
    states = []
    for i in range(0, self.sizeY):
      for j in range(0, self.sizeX):
        if not (i,j) in self.corners:
          states.append((i,j))

    return states

  def setCost(self,costType):
    self.costType = costType
    self.costX, self.costY = cost.picker(costType, self, self.sizeX, self.sizeY)
    self.heapify()
    #print self.costY
    cType = "bit"
    tc = []
    for i in range(0,self.sizeY):
      for j in range(0,self.sizeX):
        if (i,j) not in self.corners:
          if j < self.sizeX-1 and (i,j+1) not in self.corners:
            edge = (self.states[(i,j)],self.states[(i,j+1)])
            tc.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))
          if i < self.sizeY-1 and (i+1,j) not in self.corners:
            edge = (self.states[(i,j)],self.states[(i+1,j)])
            tc.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))

    sx = self.sizeX
    sy = self.sizeY
    self.totalInternalCost = tc[:]

    for i in range(0,sx):
      edge = (self.blank,self.states[(0,i)])
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
      edge = (self.states[(sy-1,i)],self.blank)
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
    for i in range(0,sy):
      edge = (self.states[(i,sx-1)],self.blank)
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))
      edge = (self.blank,self.states[(i,0)])
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))

    self.totalCost = tc
    #print self.costX, "\n\n"
    #print self.heapX

  def output(self, im = None): # this is necessary because there seems to be a bug in the PIL show function
    if im == None:
      im = self.img
    image = im.transpose(Image.FLIP_TOP_BOTTOM)
    plt.imshow(image)
    plt.show()

  def calcCostMat(self, mat, internal = False): # calculated cost given matrix of edge positions
    ct = []
    cType = "bit"
    for row in range(len(mat)):
      #print row, mat[row]
      if not internal:
        edge = (self.states[mat[row][len(mat[row]) - 1][1]],self.blank) 
        #self.output(edge[0])
        #self.output(edge[1])
        #print cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY)
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))
        edge = (self.blank,self.states[mat[row][0][0]])
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))

      for col in range(len(mat[row])):
        if not internal:
          if row == 0:
            edge = (self.blank,self.states[mat[0][col][1]])
            ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
            if col == 0:
              edge = (self.blank,self.states[mat[0][col][0]])
              ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
          elif row == len(mat) - 1:
            edge = (self.states[mat[len(mat)-1][col][1]],self.blank)
            ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
            if col == 0:
              edge = (self.states[mat[len(mat)-1][col][0]],self.blank)
              ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))

        edge = (self.states[mat[row][col][0]],self.states[mat[row][col][1]])
        ct += [cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY)]
        if row + 1 < len(mat) and col < len(mat[row+1]):
          edge = (self.states[mat[row][col][0]],self.states[mat[row+1][col][0]])
          ct += [cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY)]
      if row + 1 < len(mat) and col < len(mat[row+1]):
        edge = (self.states[mat[row][col][1]],self.states[mat[row+1][col][1]])
        ct += [cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY)]

    #assert len(ct) == 2*self.sizeX*self.sizeY - self.sizeX - self.sizeY
    return ct

  def calcCostList(self, positions, internal = False): # calculates cost given adjacency list
    ct  = []
    left = []
    right = []
    up = []
    down = []
    cType = "bit"
    revPos = dict([(v,k) for (k,v) in positions.items()])
    #print positions
    #print revPos
    for (state,(y,x)) in positions.items():
      if (y-1,x) not in revPos:
        up.append(state)
      #else:
      #  edge = (self.states[revPos[(y-1,x)]],self.states[state])
      #  ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
      if (y+1,x) not in revPos:
        down.append(state)
      else:
        edge = (self.states[state],self.states[revPos[(y+1,x)]])
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", False, self.blank, self.sizeX, self.sizeY))
      if (y,x-1) not in revPos:
        left.append(state)
      #else:
      #  edge = (self.states[revPos[(y,x-1)]],self.states[state])
      #  ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))
      if (y,x+1) not in revPos:
        right.append(state)
      else:
        edge = (self.states[state],self.states[revPos[(y,x+1)]])
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", False, self.blank, self.sizeX, self.sizeY))
    
    #print left, right, up, down
    if not internal:
      for e in left:
        ct.append(cost.indivPicker(cType, self.blank, self.states[e], "x", False, self.blank, self.sizeX, self.sizeY))
      for e in right:
        ct.append(cost.indivPicker(cType, self.states[e], self.blank, "x", False, self.blank, self.sizeX, self.sizeY))
      for e in up:
        ct.append(cost.indivPicker(cType, self.blank, self.states[e], "y", False, self.blank, self.sizeX, self.sizeY))
      for e in down:
        ct.append(cost.indivPicker(cType, self.states[e], self.blank, "y", False, self.blank, self.sizeX, self.sizeY))

    return ct

  def heapify(self): # remakes the heaps
    self.heapX = map(lambda (p,score): (score,p), self.costX.items())
    self.heapY = map(lambda (p,score): (score,p), self.costY.items())
    hq.heapify(self.heapX)
    hq.heapify(self.heapY)

  def extractHeap(self, edgesX, edgesY = None): # extracts the heaps based on given sets
    if edgesY == None:
      edgesY = edgesX
    hX = map(lambda (e,score): (score,e), filter(lambda (e,score): e in edgesX, self.costX.items()))
    hY = map(lambda (e,score): (score,e), filter(lambda (e,score): e in edgesY, self.costY.items()))
    hq.heapify(hX)
    hq.heapify(hY)
    return (hX,hY)

  def getBlanks(self): # generates the lists of blank edges surrounding the image
    self.getBlankRight()
    self.getBlankLeft()
    self.getBlankDown()
    self.getBlankUp()

  def getBlankRight(self):
    self.blankRight = []
    for i in range(self.sizeY):
      self.blankRight.append((i,self.sizeX-1))

  def getBlankLeft(self):
    self.blankLeft = []
    for i in range(self.sizeY):
      self.blankLeft.append((i,0))

  def getBlankUp(self):
    self.blankUp = []
    for i in range(self.sizeX):
      self.blankUp.append((0,i))

  def getBlankDown(self):
    self.blankDown = []
    for i in range(self.sizeX):
      self.blankDown.append((self.sizeY-1,i))

class GaussianPage(): # add proper handling for surrounding whitespace
  def __init__(self,sx,sy,ratio, noise=True):
    self.sizeX = sx
    self.sizeY = sy
    self.height = 10.0
    self.width = self.height * ratio
    self.costX = {}
    self.costY = {}
    self.heapX = {}
    self.heapY = {}
    self.totalCost = []
    if noise:
      self.noise = 0.5
      self.stdProp = 50.0
    else:
      self.noise = 0
      self.stdProp = 1000.0
    self.corners = [(-1,-1),(sy,-1),(-1,sx),(sy,sx)]
    for i in range(-1,sy+1):
      for j in range(-1,sx+1):
        for a in range(-1,sy+1):
          for b in range(-1,sx+1):
            if (i,j) != (a,b) and not (i,j) in self.corners and not (a,b) in self.corners:
                self.costX[((i,j),(a,b))] = self.gaussianCostX((i,j),(a,b))
                self.costY[((i,j),(a,b))] = self.gaussianCostY((i,j),(a,b))

    self.heapify()
    self.getBlanks()

  def getAllStates(self): # returns a list of all (i,j) positions considered, including blanks
    states = []
    for i in range(-1, self.sizeY+1):
      for j in range(-1, self.sizeX+1):
        if not (i,j) in self.corners:
          states.append((i,j))

    return states

  def gaussianCostX(self, a, b):
    if a == b:
      return float("inf")
    meanCor = self.height
    sdevCor = meanCor / self.stdProp
    meanWrong = self.noise * self.height / 2
    sdevWrong = meanWrong / self.stdProp
    errorRate = self.noise * meanCor / 5
    maxScore = meanCor + 3*sdevCor
    if a[1] == b[1] - 1 and a[0] == b[0]:
      if a[1] == -1 or b[1] == self.sizeX or (a[0] == -1 and b[0] == -1) or (a[0] == self.sizeY and b[0] == self.sizeY):
        rez = 0
        #print "x", a, b
      else:
        rez = maxScore - (r.gauss(meanCor, sdevCor) - self.sampleError(errorRate))
      self.totalCost += [rez]
      return rez
    else:
      if (a[0] == -1 and b[0] == -1) or (a[0] == self.sizeY and b[0] == self.sizeY):
        rez = 0
      else:
        rez = maxScore - (r.gauss(meanWrong, sdevWrong) + self.sampleError(errorRate))
      return rez

  def gaussianCostY(self, a, b):
    if a == b:
      return float("inf")
    meanCor = self.height # initially self.height, but causes problems without normalization as all Y costs will be smaller, so they will be favoured
    sdevCor = meanCor/ self.stdProp
    meanWrong = self.noise * self.height / 2
    sdevWrong = meanWrong / self.stdProp
    errorRate = self.noise * meanCor / 5
    maxScore = meanCor + 3*sdevCor
    if a[0] == b[0] - 1 and a[1] == b[1]:
      if a[0] == -1 or b[0] == self.sizeY or (a[1] == -1 and b[1] == -1) or (a[1] == self.sizeX and b[1] == self.sizeX):
        rez = 0
        #print "y", a, b
      else:
        rez = maxScore - (r.gauss(meanCor, sdevCor) - self.sampleError(errorRate))
      self.totalCost += [rez] 
      return rez
    else:
      if (a[1] == -1 and b[1] == -1) or (a[1] == self.sizeX and b[1] == self.sizeX):
        rez = 0
      else:
        rez = maxScore - (r.gauss(meanWrong, sdevWrong) + self.sampleError(errorRate))
      return rez

  def sampleError(self, x):
    return r.random()*x

  def calcCostMat(self, mat): # calculated cost given matrix of edge positions
    cost = []
    for row in range(len(mat)):
      for col in range(len(mat[row])):
        cost += [self.costX[(mat[row][col])]]
        if row + 1 < len(mat) and col < len(mat[row+1]):
          cost += [self.costY[(mat[row][col][0],mat[row+1][col][0])] ]
      if row + 1 < len(mat) and col < len(mat[row+1]):
        cost += [self.costY[(mat[row][col][1],mat[row+1][col][1])] ]

    #assert len(cost) == 2*self.sizeX*self.sizeY - self.sizeX - self.sizeY
    return cost

  def calcCostList(self, accumEdges): # calculates cost given adjacency list
    cost  = []
    for dr,edge in accumEdges:
      if dr == "x":
        cost.append(self.costX[edge])
      elif dr == "y":
        cost.append(self.costY[edge])
      else:
        assert False

    return cost

  def heapify(self): # remakes the heaps
    self.heapX = map(lambda (p,score): (score,p), self.costX.items())
    self.heapY = map(lambda (p,score): (score,p), self.costY.items())
    hq.heapify(self.heapX)
    hq.heapify(self.heapY)

  def extractHeap(self, edgesX, edgesY = None): # extracts the heaps based on given sets
    if edgesY == None:
      edgesY = edgesX
    hX = map(lambda (e,score): (score,e), filter(lambda (e,score): e in edgesX, self.costX.items()))
    hY = map(lambda (e,score): (score,e), filter(lambda (e,score): e in edgesY, self.costY.items()))
    hq.heapify(hX)
    hq.heapify(hY)
    return (hX,hY)

  def getBlanks(self): # generates the lists of blank edges surrounding the image
    self.getBlankRight()
    self.getBlankLeft()
    self.getBlankDown()
    self.getBlankUp()

  def getBlankRight(self):
    self.blankRight = []
    for i in range(self.sizeY):
      self.blankRight.append((i,self.sizeX-1))

  def getBlankLeft(self):
    self.blankLeft = []
    for i in range(self.sizeY):
      self.blankLeft.append((i,0))

  def getBlankUp(self):
    self.blankUp = []
    for i in range(self.sizeX):
      self.blankUp.append((0,i))

  def getBlankDown(self):
    self.blankDown = []
    for i in range(self.sizeX):
      self.blankDown.append((self.sizeY-1,i))

  def __str__(self):
    return "totalCost = " + str(self.totalCost) + "\ncostX =\n" +str([(str(x)[:5],y) for (x,y) in sorted(self.heapX, key=lambda x: x[1])]) + "\n\ncostY =\n" + str([(str(x)[:5],y) for (x,y) in sorted(self.heapY, key=lambda x: x[1])])
