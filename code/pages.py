from PIL import Image, ImageDraw, ImageFont
#import matplotlib.pyplot as plt
import random as r
import heapq as hq
import cost

neighR = [0,0,-1,1]
neighC = [1,-1,0,0]
exNeighR = [0,0,-1,1,1,1,-1,-1]
exNeighC = [1,-1,0,0,1,-1,1,-1]

class ImagePage():
  def __init__(self,sx, sy, img, cType = None, multImg = False):
    pieces = {}
    dp = {}
    rdp = {}

    if multImg:
      w, h = img[0][0].size
      for i in range(len(img)):
        for j in range(len(img[i])):
          pieces[(i,j)] = img[i][j]
          dp[(i,j)] = list(img[i][j].getdata())
          rdp[(i,j)] = list(img[i][j].rotate(90).getdata())
    else:
      width,height = img.size
      #print w, h
      h = height / sy
      w = width / sx

      img = img.crop((0,0,w*sx,h*sy))
      width,height = img.size

      data = list(img.getdata())
      row = -1
      col = -1
      cr = 0
      stt = set()
      for i in range(len(data)):
        if (i%width) % w == 0:
          col += 1
        if i % width == 0:
          col = 0
          if cr % h == 0:
            cr = 0
            row += 1
          cr += 1
        #stt.add((row, col))
        try:
          dp[(row,col)].append(data[i])
        except:
          dp[(row,col)] = [data[i]]
      #print stt

      rotData = list(img.rotate(90).getdata())
      row = -1
      col = sx
      cr = 0
      qqq = []
      for i in range(len(rotData)):
        if (i%height) % h == 0:
          #print ((row,col))
          row += 1
        if i % height == 0:
          row = 0
          if cr % w == 0:
            cr = 0
            col -= 1
          cr += 1
        stt.add((row, col))
        try:
          rdp[(row,col)].append(rotData[i])
        except:
          rdp[(row,col)] = [rotData[i]]
      #print stt
      #print qqq

      for i in range(sy):
        for j in range(sx):
          g = img.crop((j*w,i*h,(j+1)*w,(i+1)*h))
          #g.save("fffuu" + str(j), "JPEG")
          pieces[(i,j)] = g
      #print pieces
      #output(pieces[(0,1)])
      #blanks = []
      #for i in range(0,sx):
      #  pieces[(-1,i)] = blank
      #  pieces[(sy,i)] = blank
      #  blanks += [(-1,i),(sy,i)]
      #for i in range(0,sy):
      #  pieces[(i,-1)] = blank
      #  pieces[(i,sx)] = blank
      #  blanks += [(i,-1),(i,sx)]
      self.orig = img
      self.data = data
      self.data2D = []
      for i in range(len(data)/width):
        self.data2D.append(data[width*i:width*(i+1)])

    self.corners = [(-1,-1),(sy,-1),(-1,sx),(sy,sx)]
    self.sizeX = sx
    self.sizeY = sy
    blank = Image.new("1", (w,h), 255)

    self.pHeight = h
    self.pWidth = w

    self.dataPieces = dp
    self.rotDataPieces = rdp

    self.blankPos = (-42,-42)
    self.blank = blank
    self.blankCount = 1
    self.states = pieces
    if cType != None:
      self.setCost(cType)
    self.getWhitePieces()
    self.cType = "blackGaus"
    self.getBlanks()

  def verifyData(self):
    for row in range(self.sizeY):
      for col in range(self.sizeX):
        if self.dataPieces[(row,col)] != list(self.states[(row,col)].getdata()):
          raise Exception("Data verification failed at: " + str((row,col)))
        if self.rotDataPieces[(row,col)] != list(self.states[(row,col)].rotate(90).getdata()):
          raise Exception("Data verification failed at: " + str((row,col)))

    print "Data OK"
    return True

  def confEdges(self, nr):
    count = 0
    for n in self.states:
      (w,h) = self.states[n].size
      row1 = self.dataPieces[n][-w:]
      row2 = self.dataPieces[n][:w]
      col1 = self.rotDataPieces[n][:h]
      col2 = self.rotDataPieces[n][-h:]
      if len(filter(lambda x: x!=255, row1)) <= nr and len(filter(lambda x: x!=255, row2)) <= nr and len(filter(lambda x: x!=255, col1)) <= nr and len(filter(lambda x: x!=255, col2)) <= nr:
        count += 1
    return float(count) / len(self.states)

  def getWhitePieces(self, x=3): # return those pieces which have less than X black pixels on each edge
    whites = set()    
    whiteWidth = 255 * (self.pWidth - x)
    whiteHeight = 255 * (self.pHeight - x)
    for p in self.states:
      firstRow = self.dataPieces[p][:self.pWidth]
      lastRow = self.dataPieces[p][-self.pWidth:]
      firstCol = self.rotDataPieces[p][:self.pHeight]
      lastCol = self.rotDataPieces[p][-self.pHeight:]
      #print p, sum(firstRow) - whiteWidth, sum(lastRow) - whiteWidth, sum(firstCol) - whiteHeight, sum(lastCol) - whiteHeight
      if sum(firstRow) > whiteWidth and sum(lastRow) > whiteWidth and sum(firstCol) > whiteHeight and sum(lastCol) > whiteHeight:
        whites.add(p)
        #self.states[p].save("whites/" + str(p),"JPEG")

    self.whites = whites

  def pieceDistX(self, p):
    return dict(filter(lambda ((a,b),y): a == p, self.costX.items()))

  def pieceDistY(self, p):
    return dict(filter(lambda ((a,b),y): a == p, self.costY.items()))

  def getAllStates(self): # returns a list of all (i,j) positions considered, including blanks
    states = []
    for i in range(0, self.sizeY):
      for j in range(0, self.sizeX):
        if not (i,j) in self.corners:
          states.append((i,j))

    return states

  def setCost(self,costType, perX = None, perY = None, prior = None, dt = None, process = False):
    self.prx = perX
    self.pry = perY
    self.prior = prior

    self.costType = costType

    self.states[self.blankPos] = self.blank
    self.blankCount = self.sizeX * 2 + self.sizeY * 2
    print "blankCount", self.blankCount

    self.dataPieces[self.blankPos] = list(self.blank.getdata())
    self.rotDataPieces[self.blankPos] = list(self.blank.getdata())

    self.costX, self.costY = cost.picker(costType, self, perX, perY, prior, dt)



   # if process and False:
   #   self.costX = cost.normalizeCost(self.costX)
   #   self.costY = cost.normalizeCost(self.costY)

    if process:
      self.costX = cost.processCostX(self)
      self.costY = cost.processCostY(self)

    self.heapify()

    self.edgeScoreX = {}  # Used to speed up extract Heap method
    for (e, c) in self.costX.items():
       self.edgeScoreX[e] = c

    self.edgeScoreY = {}
    for (e, c) in self.costY.items():
      self.edgeScoreY[e] = c

    
    #while len(self.heapX) > 0:
    #  print hq.heappop(self.heapX)

    #print self.costY
    cType = self.cType # for evaluation purposes
    tc = []
    for i in range(0,self.sizeY):
      for j in range(0,self.sizeX):
        if (i,j) not in self.corners:
          if j < self.sizeX-1 and (i,j+1) not in self.corners:
            edge = ((i,j),(i,j+1))
            tc.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))
          if i < self.sizeY-1 and (i+1,j) not in self.corners:
            edge = ((i,j),(i+1,j))
            tc.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))

    sx = self.sizeX
    sy = self.sizeY
    self.totalInternalCost = tc[:]

    for i in range(0,sx):
      edge = (self.blankPos,(0,i))
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
      edge = ((sy-1,i),self.blankPos)
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
    for i in range(0,sy):
      edge = ((i,sx-1),self.blankPos)
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))
      edge = (self.blankPos,(i,0))
      tc.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))

    self.totalCost = tc
    #print self.costX, "\n\n"
    #print self.heapX

  def output(self, im = None): # this is necessary because there seems to be a bug in the PIL show function
    if im == None:
      im = self.img
    image = im.transpose(Image.FLIP_TOP_BOTTOM)
    plt.imshow(image)
    plt.show()

  def calcGroups(self, pos): # calculates sizes of correct groups given positions
    revPos = dict([(v,k) for (k,v) in pos.items()])
    groups = {}
    gs = 0

    nX = [0,0,1,-1]
    nY = [1,-1,0,0]

    while len(revPos) > 0:
      curInd = min(revPos.keys())
      groups[gs] = {}

      while curInd != None:
        curPiece = revPos[curInd]
        del revPos[curInd]
        groups[gs][curPiece] = curInd

        nextInd = None
        for i in range(4):
          potNextInd = (curInd[0] + nY[i], curInd[1] + nX[i])
          if potNextInd in revPos:
            if curPiece[0] + nY[i] == revPos[potNextInd][0] and curPiece[1] + nX[i] == revPos[potNextInd][1]:
              nextInd = potNextInd

        curInd = nextInd
      gs += 1

    assert(sum(map(len,groups.values())) == len(pos))
    self.vizPos(groups, fl="groups" + str(self.sizeX), multiple = True)

    return map(len,groups.values()) 

  def calcCostMat(self, mat, internal = False): # calculated cost given matrix of edge positions
    ct = []
    cType = self.cType
    for row in range(len(mat)):
      #print row, mat[row]
      if not internal:
        edge = (mat[row][len(mat[row]) - 1][1], self.blankPos) 
        #self.output(edge[0])
        #self.output(edge[1])
        #print cost.indivPicker(cType, edge[0], edge[1], "x", self, False)
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))
        #print "x edge", mat[row][len(mat[row]) - 1][1]
        edge = (self.blankPos, mat[row][0][0])
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))
        #print "x edge", mat[row][0][0]

      for col in range(len(mat[row])):
        if not internal:
          if row == 0:
            edge = (self.blankPos, mat[0][col][1])
            #print "y edge", mat[0][col][1]
            ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
            if col == 0:
              edge = (self.blankPos, mat[0][col][0])
              #print "y edge", mat[0][col][0]
              ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
          elif row == len(mat) - 1:
            edge = (mat[len(mat)-1][col][1], self.blankPos)
            ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
            #print "y edge", mat[len(mat)-1][col][1]
            if col == 0:
              edge = (mat[len(mat)-1][col][0], self.blankPos)
              ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
              #print "y edge", mat[len(mat)-1][col][0]
          if row != len(mat)-1 and col >= len(mat[row+1]):
            edge = (mat[row][col][1], self.blankPos)
            ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
            #print "y edge", mat[row][col][1]
          if row != 0 and col >= len(mat[row-1]):
            edge = (self.blankPos, mat[row][col][1])
            ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
            #print "y edge", mat[row][col][1]
            

        edge = (mat[row][col][0], mat[row][col][1])
        ct += [cost.indivPicker(cType, edge[0], edge[1], "x", self, False)]
        
        if row + 1 < len(mat) and col <= len(mat[row+1]):
          if col == len(mat[row+1]):
            edge = (mat[row][col][0],mat[row+1][col-1][1])
          else:
            edge = (mat[row][col][0], mat[row+1][col][0])
          ct += [cost.indivPicker(cType, edge[0], edge[1], "y", self, False)]
          
      col = len(mat[row]) - 1
      if row + 1 < len(mat) and col < len(mat[row+1]):
        edge = (mat[row][col][1], mat[row+1][col][1])
        ct += [cost.indivPicker(cType, edge[0], edge[1], "y", self, False)]
        

    #assert len(ct) == 2*self.sizeX*self.sizeY - self.sizeX - self.sizeY
    return ct

  def calcCostList(self, positions, internal = False): # calculates cost given dictionary of edge positions
    ct  = []
    left = []
    right = []
    up = []
    down = []
    cType = self.cType
    revPos = dict([(v,k) for (k,v) in positions.items()])
    #print positions
    #print revPos
    for (state,(y,x)) in positions.items():
      if (y-1,x) not in revPos:
        up.append(state)
      #else:
      #  edge = (revPos[(y-1,x)], state)
      #  ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
      if (y+1,x) not in revPos:
        down.append(state)
      else:
        edge = (state, revPos[(y+1,x)])
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "y", self, False))
      if (y,x-1) not in revPos:
        left.append(state)
      #else:
      #  edge = (revPos[(y,x-1)], state)
      #  ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))
      if (y,x+1) not in revPos:
        right.append(state)
      else:
        edge = (state, revPos[(y,x+1)])
        ct.append(cost.indivPicker(cType, edge[0], edge[1], "x", self, False))
    
    #print "x edges", left, right
    #print "y edges", up, down
    if not internal:
      for e in left:
        ct.append(cost.indivPicker(cType, self.blankPos, e, "x", self, False))
      for e in right:
        ct.append(cost.indivPicker(cType, e, self.blankPos, "x", self, False))
      for e in up:
        ct.append(cost.indivPicker(cType, self.blankPos, e, "y", self, False))
      for e in down:
        ct.append(cost.indivPicker(cType, e, self.blankPos, "y", self, False))

    return ct

  def calcCorrectEdges(self, positions, ignoreWhites = False): # percent of edges that are correct, only internal
    totalEdges = (self.sizeX - 1) * self.sizeY + (self.sizeY - 1) * self.sizeX    
    
    if ignoreWhites:
      for (r,c) in self.whites:
        if r == 0 and c == 0:
          totalEdges -= 2
        elif r == 0 or c == 0 or r == self.sizeY -1 or c == r.sizeX -1:
          totalEdges -= 3
        else:
          totalEdges -= 4

    revPos = dict([(v,k) for (k,v) in positions.items()])
    correctEdges = 0.0
    for ((sy,sx),(y,x)) in positions.items():
      if (y+1,x) in revPos and revPos[(y+1,x)] == (sy+1,sx):
        correctEdges += 1
      if (y,x+1) in revPos and revPos[(y,x+1)] == (sy,sx+1):
        correctEdges += 1

    return correctEdges / totalEdges

  def heapify(self): # remakes the heaps
    self.heapX = map(lambda (p,score): (score,p), self.costX.items())
    self.heapY = map(lambda (p,score): (score,p), self.costY.items())
    hq.heapify(self.heapX)
    hq.heapify(self.heapY)

  def extractHeap(self, edgesX, edgesY = None): # extracts the heaps based on given sets
    if edgesY == None:
      edgesY = edgesX
    hX = reduce(lambda l1, l2 : l1 + l2,  map(lambda x: [(self.edgeScoreX[x],x)], edgesX), [])
    hY = reduce(lambda l1, l2 : l1 + l2,  map(lambda y: [(self.edgeScoreY[y],y)], edgesY), [])
    hq.heapify(hX)
    hq.heapify(hY)
    return (hX,hY)

  def oldExtractHeap(self, edgesX, edgesY = None): # extracts the heaps based on given sets
    if edgesY == None:
      edgesY = edgesX
    hX = map(lambda (e,score): (score,e), filter(lambda (e,score): e in edgesX, self.costX.items()))
    hY = map(lambda (e,score): (score,e), filter(lambda (e,score): e in edgesY, self.costY.items()))
    hq.heapify(hX)
    hq.heapify(hY)
    return (hX,hY)

  def verifyExtractHeap(self, edgesX, edgesY = None):
    return self.extractHeap(edgesX, edgesY) == self.oldExtractHeap(edgesX, edgesY)

  def getBoxes(self, pixels = None):
    if pixels == None:
      pixels = self.data2D

    print '\n'.join(map(''.join, map(lambda x: map(lambda y:str(y/255), x), pixels)))
    curX = 0
    curY = 0
    boxes = []
    seen = set()
    
    while True:
      print curY, curX
      start = None
      while curY < len(pixels):
        curX = 0
        while curX < len(pixels[curY]):
          if pixels[curY][curX] == 0 and (curY,curX) not in seen:
            start = (curY, curX)
            break
          curX += 1
        if start != None:
          break
        curY += 1

      if start == None:
        break

      seen.add(start)
      end = start
      contains = [start]
      cc = 0
      while cc < len(contains):
        cur = contains[cc]
        cc += 1
        for i in range(8):
          pot = (cur[0] + exNeighR[i], cur[1] + exNeighC[i])
          if pot not in contains and pot[0] >= 0 and pot[1] >= 0 and pot[0] < len(pixels) and pot[1] < len(pixels[pot[0]]) and pixels[pot[0]][pot[1]] == 0:
            contains.append(pot)
            seen.add(pot)
            if pot[0] > end[0]:
              end = (pot[0], end[1])
            if pot[1] > end[1]:
              end = (end[0], pot[1])
            if pot[0] < start[0]:
              start = (pot[0], start[1])
            if pot[1] < start[1]:
              start = (start[0], pot[1])

      boxes.append((start,end))

    print boxes
    print sorted(seen)
    return boxes

  def getBigRows(self, pixels = None):
    if pixels == None:
      pixels = self.data2D

    curY = 0
    curX = 0
    rows = []

    while True:
      start = None
      while curY < len(pixels):
        curX = 0
        while curX < len(pixels[curY]):
          if pixels[curY][curX] == 0:
            start = curY
            break
          curX += 1
        curY += 1
        if start != None:
          break
      if start == None:
        break
      while curY < len(pixels):
        curX = 0
        cont = False
        while curX < len(pixels[curY]):
          if pixels[curY][curX] == 0:
            cont = True
            break
          curX += 1
        curY += 1
        if not cont:
          break
      rows.append((start, curY))

    return rows

  def getRows(self, pixels = None):
    if pixels == None:
      pixels = self.data2D

    bigRows = self.getBigRows(pixels)
    boxes = self.getBoxes(pixels)

    rowTops = {}
    rowBottoms = {}
    heights = {}
    for (start, end) in boxes:
      width = end[1] - start[1]
      height = end[0] - start[0]
      try:
        heights[height] += 1
      except:
        heights[height] = 0

      if start[0] not in rowTops:
        rowTops[start[0]] = 0
      else:
        rowTops[start[0]] += width

      if end[0] not in rowBottoms:
        rowBottoms[end[0]] = 0
      else:
        rowBottoms[end[0]] += width

    orderedTops = sorted(rowTops.items(), key=lambda x:x[1], reverse=True)
    orderedBottoms = sorted(rowBottoms.items(), key=lambda x:x[1], reverse=True)
    #modeHeight = sorted(heights.items(), key=lambda x:x[1], reverse=True)[0][0]

    finTops = {}
    finBottoms = {}

    for (topY, topV) in orderedTops:
      for (start, end) in bigRows:
        if topY >= start and topY <= end and (start, end) not in finTops:
          finTops[(start, end)] = topY

    for (bottomY, bottomV) in orderedBottoms:
      for (start, end) in bigRows:
        if bottomY >= start and bottomY <= end and (start, end) not in finBottoms:
          finBottoms[(start, end)] = bottomY

    smallRows = []
    for key in bigRows:
      smallRows.append((finTops[key], finBottoms[key]))

    return (bigRows, smallRows)

  def getRows1(self, pixels = None):
    if pixels == None:
      pixels = self.data2D

    bigRows = self.getBigRows(pixels)

    rowTops = {}
    rowBottoms = {}
    heights = {}
    
    curY = 0
    curX = 0
    prevRow = 0
    curRow = -1000
    nextRow = 0
    while curY < len(pixels):
      nextRow = 0
      curX = 0
      while curX < len(pixels[curY]):
        if pixels[curY][curX] == 0:
          nextRow += 1
        curX += 1

      posTop = curRow - prevRow
      posBot = curRow - nextRow
      if posTop > posBot and posTop > 0:
        rowTops[curY] = posTop

      if posBot > posTop and posBot > 0:
        rowBottoms[curY] = posBot
      prevRow = curRow
      if prevRow == -1000:
        prevRow = 0
      curRow = nextRow
      curY += 1

    orderedTops = sorted(rowTops.items(), key=lambda x:x[1], reverse=True)
    orderedBottoms = sorted(rowBottoms.items(), key=lambda x:x[1], reverse=True)
    #modeHeight = sorted(heights.items(), key=lambda x:x[1], reverse=True)[0][0]

    finTops = {}
    finBottoms = {}

    for (topY, topV) in orderedTops:
      for (start, end) in bigRows:
        if topY >= start and topY <= end - (end-start)/2 and (start, end) not in finTops:
          finTops[(start, end)] = topY

    for (bottomY, bottomV) in orderedBottoms:
      for (start, end) in bigRows:
        if bottomY >= start + (end-start)/2 and bottomY <= end and (start, end) not in finBottoms:
          finBottoms[(start, end)] = bottomY

    smallRows = []
    for key in bigRows:
      if key not in finTops:
        finTops[key] = key[0]
      if key not in finBottoms:
        finBottoms[key] = key[1]
      smallRows.append((finTops[key], finBottoms[key]))

    return (bigRows, smallRows)

  def calcPiecesRows(self):
    self.piecesSmallRows = {(-42,-42):[]}
    self.piecesBigRows = {(-42,-42):[]}
    for row in range(self.sizeY):
      for col in range(self.sizeX):
        data = self.dataPieces[(row,col)]
        data2D = []
        for i in range(len(data)/self.pWidth):
          data2D.append(data[self.pWidth*i:self.pWidth*(i+1)])
        self.piecesBigRows[(row,col)], self.piecesSmallRows[(row,col)] = self.getRows1(data2D)

  def calcOrients(self):
    orient = {}
    white = 0
    for row in range(self.sizeY):
      for col in range(self.sizeX):
        piece = (row,col)
        small = self.piecesSmallRows[piece]
        big = self.piecesBigRows[piece]
        assert (len(small) == len(big))

        top = 0
        bottom = 0
        data = self.dataPieces[piece]
        if not 0 in data:
          white += 1
          continue

        data2D = []
        for i in range(len(data)/self.pWidth):
          data2D.append(data[self.pWidth*i:self.pWidth*(i+1)])

        for c in range(len(small)):
          curY = big[c][0]
          while curY < small[c][0]:

            curX = 0
            while curX < self.pWidth:
              if curY >= 0 and curY < len(data2D) and data2D[curY][curX] == 0:
                top += 1
              curX += 1
            curY += 1

          curY = small[c][1] + 1
          while curY <= big[c][1]:
            curX = 0
            while curX < self.pWidth:
              if curY >= 0 and curY < len(data2D) and data2D[curY][curX] == 0:
                bottom += 1
              curX += 1
            curY += 1

        orient[piece] = (top, bottom)

    upside = filter(lambda (k,(x,y)): x<y, orient.items())
    unknown = filter(lambda (k,(x,y)): x==y, orient.items())

    """
    if upside != []:
      print upside[0]
      piece = upside[0][0]
      im = self.states[piece]
      #im.show()
      final = Image.new("RGB", im.size)
      drawFin = ImageDraw.Draw(final)
      final.paste(im, (0,0))
      for (start,end) in self.piecesBigRows[piece]:
        drawFin.line([(0,start), (im.size[0],start)], fill=(255,0,0))
        drawFin.line([(0,end), (im.size[0],end)], fill=(255,0,255))

      for (start,end) in self.piecesSmallRows[piece]:
        drawFin.line([(0,start), (im.size[0],start)], fill=(0,255,0))
        drawFin.line([(0,end), (im.size[0],end)], fill=(0,0,255))
      final.save("mafffff","JPEG")
      raw_input("qqqq")
      """
    return white, len(unknown), len(upside)

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

  def vizPos(self, poses, fl=None, multiple=False, rows=False):

    if fl == None:
      fl = "temp"
  
    red = Image.new("RGB", (self.pWidth,self.pHeight), (210,0,0))
    font = ImageFont.truetype("wendy.ttf", 15)

    extraPixels = 10

    backs = []
    finalH = 0
    finalW = 0

    if not multiple:
      poses = {1:poses}

    for (k, pos) in poses.items():
      reds = set()
      for (k, (pr,pc)) in pos.items():
        for i in range(4):
          cr = pr + neighR[i]
          cc = pc + neighC[i]
          if (cr,cc) not in pos.values():
            reds.add((cr,cc))

      revPos = sorted(pos.items(), key=lambda x: x[1])
      
      minY = revPos[0][1][0]
      maxY = revPos[-1][1][0]

      minX = revPos[0][1][1]
      maxX = revPos[0][1][1]
      for (node, pos) in revPos:
        if pos[1] < minX:
          minX = pos[1]
        if pos[1] > maxX:
          maxX = pos[1]

      back = Image.new("RGB",((maxX - minX + 1)*self.pWidth + extraPixels, (maxY - minY + 1)*self.pHeight + extraPixels) )

      for (node, (curY, curX)) in revPos:
        piece = self.states[node].convert("RGB").copy()
        drawPiece = ImageDraw.Draw(piece)
        drawPiece.text((2,2),str(node), fill=(255,0,0), font=font)
        pw, ph = piece.size
        drawPiece.line([(0,0),(pw-1,0),(pw-1,ph-1),(0,ph-1),(0,0)], fill=(255,0,0))
        if rows:
          smallRows =self.piecesSmallRows[node]
          for (start,end) in smallRows:
            drawPiece.line([(0,start), (self.pWidth,start)], fill=(255,0,0))
            drawPiece.line([(0,end), (self.pWidth,end)], fill=(255,0,0))

        back.paste(piece, ((curX-minX + 0)*self.pWidth, (curY-minY + 0)*self.pHeight))

      #for (curY, curX) in reds:
      #  back.paste(red, ((curX-minX + 1)*self.pWidth, (curY-minY + 1)*self.pHeight)) 

      bw, bh = back.size
      finalW += bw
      if bh > finalH:
        finalH = bh

      backs.append(back)

    final = Image.new("RGB", (finalW, finalH))
    curW = 0
    curH = 0
    for back in backs:
      bw,bh = back.size
      final.paste(back, (curW, curH))
      curW += bw
  
    final.save(fl,"JPEG")

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
