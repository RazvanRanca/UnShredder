import sys
from matplotlib import pyplot as plt
import search
import cost
import verifCost
import pages
import time
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import math
from scipy.stats import mode
import os
import re

class DNode:
  def __init__(self, tp = None, probB = None, feat = None, b = None, w = None):
    self.feature = feat
    self.type = tp
    if probB == None:
      self.isLeaf = False
      self.probBlack = None
      self.probWhite = None
    else:
      self.isLeaf = True
      self.probBlack = probB
      self.probWhite = 1 - probB

    self.whiteChild = w
    self.blackChild = b

  def getProb(self, data, tind, tval):
    if self.isLeaf:
      if tval == 0:
        return self.probBlack
      else:
        return self.probWhite
    else:
      if self.type == "V":
        r = tind + self.feature[0]
        c = len(data[r]) + self.feature[1]

      if self.type == "H":
        r = len(data) + self.feature[0]
        c = tind + self.feature[1]
      try:
        if data[r][c] == 0:
          return self.blackChild.getProb(data, tind, tval)
        else:
          return self.whiteChild.getProb(data, tind, tval)
      except:
        return None

class DTree:
  def __init__(self, ims, depth=5):
    self.rootH = None
    self.rootV = None
    extentSize = 3
    contextV = []
    for f in range(-extentSize,extentSize+1):
      for s in range(-extentSize,0):
        if not (f==0 and s==0):
          contextV.append((f,s))

    contextH = []
    for f in range(-extentSize,0):
      for s in range(-extentSize,extentSize+1):
        if not (f==0 and s==0):
          contextH.append((f,s))  

    featuresV = {}
    categsV = []
    featuresH = {}
    categsH = []
    for c in contextV:
      featuresV[c] = []

    for c in contextH:
      featuresH[c] = []

    for im in ims:
      w,h = im.size
      ld = list(im.getdata())
      
      self.priorBlack = ld.count(0) / float(len(ld))
      self.priorWhite = 1 - self.priorBlack

      bd = []
      for r in range(h):
        bd.append([])
        for c in range(w):
          bd[r].append(ld[r*w + c])

      div = 255.0
      for r in range(extentSize, h -extentSize):
        for c in range(extentSize, w):
          vr = bd[r][c]/div
          categsV.append(vr)
          for i in contextV:
            nr = r+i[0]
            nc = c+i[1]
            featuresV[i].append(bd[nr][nc]/div)

      for r in range(extentSize, h):
        for c in range(extentSize, w - extentSize):
          vr = bd[r][c]/div
          categsH.append(vr)
          for i in contextH:
            nr = r+i[0]
            nc = c+i[1]
            featuresH[i].append(bd[nr][nc]/div)

    self.rootV = self.getFeature(categsV, featuresV, depth, "V")
    self.rootH = self.getFeature(categsH, featuresH, depth, "H")

  def getFeature(self, categs, features, maxDepth, cat, depth=1, pReach = 1.0):
    minEnt = 1.0
    bestFeature = None
    bestData = None

    for (k,v) in features.items():
      count = 0.0
      blacks = 0.0
      predB = 0.0
      predW = 0.0    
      for i in range(len(v)):
        ct = categs[i]
        count += 1
        if v[i] == 0:
          blacks += 1
          if ct == 0:
            predB += 1
        else:
          if ct == 1:
            predW += 1

      probB = blacks/count
      probW = 1 - probB
      whites = count - blacks

      ent = 0
      if probB > 0:
        ent += probB * getH2(predB / blacks) 
      if probW > 0:
        ent += probW * getH2(predW / whites)

      if ent < minEnt:
        minEnt = ent
        bestFeature = k
        bestData = probB, probW, predB / blacks, predW / whites

    catPredB = "B"
    predB = bestData[2]
    if predB < 0.5:
      catPredB = "W"
      predB = 1 - predB

    catPredW = "W"
    predW = bestData[3]
    if predW < 0.5:
      catPredW = "W"
      predW = 1 - predW

    probCorr = bestData[0] * predB + bestData[1] * predW

    print cat, depth, pReach, bestData, bestFeature, minEnt, catPredB, catPredW, probCorr

    if depth >= maxDepth:
      return DNode(tp = cat, feat = bestFeature, b = DNode(cat,bestData[2]), w = DNode(cat,1 - bestData[3]))

    nCatB = []
    nCatW = []
    setFeatB = set()
    nFeatB = {}
    nFeatW = {}
    for c in features.keys():
      if c == bestFeature:
        continue
      nFeatB[c] = []
      nFeatW[c] = []

    bv = features[bestFeature]
    for i in range(len(bv)):
      ct = categs[i]
      if bv[i] == 0:
        nCatB.append(ct)
        setFeatB.add(i)
      else:
        nCatW.append(ct)

    for (k,v) in features.items():
      if k == bestFeature:
        continue
      for i in range(len(v)):
        if i in setFeatB:
          nFeatB[k].append(v[i])
        else:
          nFeatW[k].append(v[i])

    pcB = self.getFeature(nCatB, nFeatB, maxDepth, cat, depth+1, pReach*bestData[0])
    pcW = self.getFeature(nCatW, nFeatW, maxDepth, cat, depth+1, pReach*bestData[1])

    return DNode(tp = cat, b = pcB, w = pcW, feat = bestFeature)

  def getProb(self, data, tind, tval, cat):
    ret = -1
    if cat == "H":
      ret = self.rootH.getProb(data, tind, tval)
    if cat == "V":
      ret = self.rootV.getProb(data, tind, tval)
    if ret == None:
      if tval == 0:
        return self.priorBlack
      else:
        return self.priorWhite

    return ret

def imgRotate(img, deg, bCol = (255,0,127,0), format="RGBA"): # rotates image clockwise
  img.save("tempRot.png","PNG")
  os.system("convert -background 'rgb" + str(bCol) +"' -rotate " + str(deg) + " tempRot.png tempRot.png")
  img = Image.open("tempRot.png").convert(format)
  return img

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

def cutShreds(sx, sy, img):
  img = img.convert('RGBA')
  margin = 20
  width,height = img.size
  h = height / sy
  w = width / sx
  pieces = []
  backCol = (255,0,127,255)

  totalW = margin
  maxTotalW = 0
  maxH = 0
  lb = sx*sy # int(math.sqrt(sx*sy))
  for i in range(sy):
    for j in range(sx):
      g = img.crop((j*w,i*h,(j+1)*w,(i+1)*h))
      gw, gh = g.size
      maxH = max(maxH, gh)
      totalW += gw + margin
      pieces.append(g)
      maxTotalW = max(maxTotalW, totalW)
      if len(pieces) % lb == 0:
        totalW = margin

  random.shuffle(pieces)
  totalH = margin + (maxH + margin)# * (len(pieces)/lb + 1)
  scrambled = Image.new("RGBA", (maxTotalW, totalH), backCol)
  ind = 0
  curW = margin
  curH = margin
  for i in range(sy):
    for j in range(sx):
      if ind % lb == 0:
        curW = margin
        curH = margin + (maxH+margin) * ind/lb
      scrambled.paste(pieces[ind], (curW, curH), pieces[ind])
      w, h = pieces[ind].size
      curW += w + margin
      ind += 1

  scrambled.save("idealShreds","PNG")

def scrambleImage(sx, sy, img):
  img = img.convert('RGBA')
  margin = 20
  width,height = img.size
  h = height / sy
  w = width / sx
  pieces = []
  backCol = (255,0,127,255)

  totalW = margin
  maxTotalW = 0
  maxH = 0
  maxRot = 10
  lb = sx*sy # int(math.sqrt(sx*sy))
  for i in range(sy):
    for j in range(sx):
      g = img.crop((j*w,i*h,(j+1)*w,(i+1)*h))
      rotAngle = random.random()*maxRot*2 - maxRot
      if random.random() > 0.5:
        rotAngle += 180
      g = g.rotate(rotAngle, Image.BICUBIC, expand=True) # imgRotate(g, rotAngle) #
      gw, gh = g.size
      maxH = max(maxH, gh)
      totalW += gw + margin
      pieces.append(g)
      maxTotalW = max(maxTotalW, totalW)
      if len(pieces) % lb == 0:
        totalW = margin

  #random.shuffle(pieces)
  totalH = margin + (maxH + margin)# * (len(pieces)/lb + 1)
  scrambled = Image.new("RGBA", (maxTotalW, totalH), backCol)
  ind = 0
  curW = margin
  curH = margin
  for i in range(sy):
    for j in range(sx):
      if ind % lb == 0:
        curW = margin
        curH = margin + (maxH+margin) * ind/lb
      scrambled.paste(pieces[ind], (curW, curH), pieces[ind])
      w, h = pieces[ind].size
      curW += w + margin
      ind += 1

  scrambled.save("scrambledRot","PNG")

def processScan(img, bCol = None):
  if bCol == None:
    bCol = img.getpixel((0,0))

  print bCol
  tolerance = 400
  minSize = 100

  neigh8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
  found = []
  maxs = []
  seen = set()
  w, h = img.size
  print w, h
  data = list(img.getdata())
  data2D = []
  for i in range(h):
    data2D.append(data[i*w:(i+1)*w])

  assert(''.join(map(str, data)) == ''.join(map(lambda x: ''.join(map(str,x)), data2D)))

  ind = 0
  for r in range(len(data2D)):
    if r % 100 == 0:
      print r
    for c in range(len(data2D[r])):
      if (r,c) in seen:
        continue
      pixel = data2D[r][c]
      diff = abs(pixel[0] - bCol[0]) + abs(pixel[1] - bCol[1]) + abs(pixel[2] - bCol[2]) + pixel[3]
      #print pixel
      if diff > tolerance:
        seen.add((r,c))
        minR = r
        minC = c
        maxR = r
        maxC = c
        curList = [(r,c)]
        cind = 0
        while cind < len(curList):
          cr, cc = curList[cind]
          cind += 1
          for (nr, nc) in neigh8:
            next = (cr + nr, cc + nc)
            if next not in seen:
              pixel = data2D[next[0]][next[1]]
              diff = abs(pixel[0] - bCol[0]) + abs(pixel[1] - bCol[1]) + abs(pixel[2] - bCol[2]) + pixel[3]
              if diff > tolerance:
                seen.add(next)
                curList.append(next)
                if next[0] < minR:
                  minR = next[0]
                if next[1] < minC:
                  minC = next[1]
                if next[0] > maxR:
                  maxR = next[0]
                if next[1] > maxC:
                  maxC = next[1]
        found.append({})
        maxs.append((maxR-minR,maxC-minC))
        #print ind, minR, minC, maxR, maxC
        for (row,col) in curList:
          found[ind][(row-minR,col-minC)] = data2D[row][col]
        ind += 1
  
  print map(len,found)
  nFound = []
  nMaxs = []
  for i in range(len(found)):
    if len(found[i]) > minSize:
      nFound.append(found[i])
      nMaxs.append(maxs[i])

  found = nFound
  maxs = nMaxs
  print len(found), "pieces detected"

  shreds = []
  lens = []
  turnPoint = []
  maxRs = []
  for ind in range(len(found)):
    shreds.append([])
    lens.append([])
    maxR = 0
    minC = float("inf")
    point = None
    for r in range(maxs[ind][0]):
      for c in range(maxs[ind][1]):
        if (r,c) not in found[ind]:
          shreds[ind].append(bCol)
        else:
          shreds[ind].append(found[ind][(r,c)])
          if (r,c-1) not in found[ind]:
            lens[ind].append((c,r))
            if c < minC:
              minC = c
              point = r
            if r > maxR:
              maxR = r
    turnPoint.append(point)
    maxRs.append(maxR)

  angles = []
  medAngles = []
  #print turnPoint
  #print maxRs
  #print maxs
  #print map(len, found)
  for ind in range(len(lens)):
    if turnPoint[ind] > maxRs[ind] / 2:
      run = lens[ind][:turnPoint[ind]]
    else:
      run = lens[ind][turnPoint[ind]:]
    curR = run[-1][1]
    curMinC = min(run)[0]
    mult = 1
    if run[0][0] - run[-1][0] < 0:
      #mult = -1
      curR = run[0][1]
    angles.append([])
    #if ind == 0:
    #  print '\n'.join(map(lambda x : str((x,float(x[0] - curMinC)/(curR - x[1]-1), math.degrees(math.atan2(x[0] - curMinC, curR - x[1])))),run))
    for i in range(len(run)):
      angles[ind].append(math.atan2(run[i][0] - curMinC, curR - run[i][1]))
    medAngles.append(math.degrees(mult*sorted(angles[ind])[len(angles[ind])/2]))


  print medAngles[0]
  print sorted(angles[0])
  #print maxRs
  #print turnPoint
  #print medAngles
  rezs = []
  minMaxW = float("inf")
  minMaxH = float("inf")
  lefts = []
  ups = []
  orients = []
  for ind in range(len(medAngles)):
    wip = Image.new("RGBA", (maxs[ind][1], maxs[ind][0]), bCol)
    wip.putdata(shreds[ind])
    #wip.save("wipInit" + str(ind),"PNG")
    wip = imgRotate(wip, -1*medAngles[ind], bCol) #  wip.rotate(medAngles[ind], Image.BICUBIC, expand=True) # 
    #wip.save("wipRot" + str(ind),"PNG")
    wipBack = Image.new("RGBA", wip.size, bCol)
    wipBack.paste(wip, None, wip)
    #wipBack.save("wipBack" + str(ind),"PNG")
    wipOrient = orient2(wipBack, str(ind))
    (left, up, maxW, maxH) = extractShred(wipOrient, bCol, str(ind))
    orients.append(wipOrient)
    lefts.append(left)
    ups.append(up)
    if maxW < minMaxW:
      minMaxW = maxW
    if maxH < minMaxH:
      minMaxH = maxH

  for ind in range(len(orients)):
    wipCropped = orients[ind].crop((lefts[ind],ups[ind],lefts[ind] + minMaxW, ups[ind] + minMaxH))
    w,h = wipCropped.size
    #wipCropped = wipCropped.resize((w/2,h/2), Image.ANTIALIAS)
    wipCropped = polarize(wipCropped, bCol)
    rezs.append(wipCropped)
    wipCropped.save("wipScan" + str(ind),"JPEG")

  return rezs

def polarize(img, col):
  data = list(img.getdata())
  threshold = 450
  #print threshold
  #print data[200:220]
  newData = map(lambda x: 0 if x[0] + x[1] + x[2] < threshold else 255, data)
  newImg = Image.new("1", img.size)
  newImg.putdata(newData)
  return newImg
 
def extractShred(img, bCol = None, name = ""): # assumes shred is at 90 degree angles
  if bCol == None:
    bCol = img.getpixel((0,0))


  tolerance = 500
  data = list(img.getdata())
  data2D = []
  w, h = img.size
  for i in range(h):
    data2D.append(data[i*w:(i+1)*w])

  minAccHeight = 10
  minAccWidth = 5

  maxWidth = 0
  maxHeight = 0
  width = 0
  height = 0

  count = 0
  widths = []
  for r in range(len(data2D)): 
    start = 0
    found = False
    for c in range(len(data2D[r])):
      pixel = data2D[r][c]
      diff = abs(pixel[0] - bCol[0]) + abs(pixel[1] - bCol[1]) + abs(pixel[2] - bCol[2]) + pixel[3]
      if diff > tolerance and start == 0:
        start = c
      if diff <= tolerance and start != 0:
        #print start, r, pixel
        width = c - start
        found = True
        break
    if found:
      widths.append(width)
      count += 1
    elif start != 0:
      widths.append(w)
      count += 1

  maxWidth = int(mode(filter(lambda w: w >= minAccWidth, widths))[0][0]) #sorted(widths)[count/2]

  lefts = []
  for r in range(len(data2D)):  
    for c in range(len(data2D[r])):
      pixel = data2D[r][c]
      diff = abs(pixel[0] - bCol[0]) + abs(pixel[1] - bCol[1]) + abs(pixel[2] - bCol[2]) + pixel[3]
      if diff > tolerance:
        lefts.append(c)
        break

  ups = []
  for c in range(len(data2D[0])):
    for r in range(len(data2D)):  
      pixel = data2D[r][c]
      diff = abs(pixel[0] - bCol[0]) + abs(pixel[1] - bCol[1]) + abs(pixel[2] - bCol[2]) + pixel[3]
      if diff > tolerance:
        ups.append(r)
        break

  left = int(mode(lefts)[0][0]) #sorted(lefts)[len(lefts)/2]
  up = int(mode(ups)[0][0]) #sorted(ups)[len(ups)/2]

  count = 0
  heights = []
  for c in range(len(data2D[0])):
    start = 0
    found = False
    for r in range(len(data2D)):
      pixel = data2D[r][c]
      diff = abs(pixel[0] - bCol[0]) + abs(pixel[1] - bCol[1]) + abs(pixel[2] - bCol[2]) + pixel[3]
      if diff > tolerance and start == 0:
        start = r
      if diff <= tolerance and start != 0:
        #print start, r, pixel
        height = r - start
        found = True
        break
    if found:
      heights.append(height)
      count += 1
    elif start != 0:
      heights.append(h)
      count += 1


  maxHeight = int(mode(filter(lambda h: h >= minAccHeight, heights))[0][0]) #sorted(heights)[count/2]

  print img.size, up, left, maxWidth, maxHeight

  #cropped = img.crop((left,up,left + maxWidth, up + maxHeight))
  #cropped.save("wipCropped" + name, "PNG")

  return (left,up,maxWidth,maxHeight)


def orient1(img): # figure out if piece is vertical or horizontal

  data = list(img.getdata())
  pixels = []
  w, h = img.size
  for i in range(h):
    pixels.append(data[i*w:(i+1)*w])

  curY = 0
  curX = 0
  rows = []
  tolerance = 250

  while True:
    start = None
    while curY < len(pixels):
      curX = 0
      while curX < len(pixels[curY]):
        pixel = pixels[curY][curX]
        diff = pixel[0] + pixel[1] + pixel[2]
        if diff < tolerance:
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
        pixel = pixels[curY][curX]
        diff = pixel[0] + pixel[1] + pixel[2]
        if diff < tolerance:
          cont = True
          break
        curX += 1
      curY += 1
      if not cont:
        break
    rows.append((start, curY))

  rotImg = imgRotate(img, 90) #img.rotate(90)
  data = list(rotImg.getdata())
  pixels = []
  w, h = rotImg.size
  for i in range(h):
    pixels.append(data[i*w:(i+1)*w])

  curY = 0
  curX = 0
  rotRows = []

  while True:
    start = None
    while curY < len(pixels):
      curX = 0
      while curX < len(pixels[curY]):
        pixel = pixels[curY][curX]
        diff = pixel[0] + pixel[1] + pixel[2]
        if diff < tolerance:
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
        pixel = pixels[curY][curX]
        diff = pixel[0] + pixel[1] + pixel[2]
        if diff < tolerance:
          cont = True
          break
        curX += 1
      curY += 1
      if not cont:
        break
    rotRows.append((start, curY))

  #print "rot90", len(rows), len(rotRows)

  if len(rows) > len(rotRows):
    return (img, rows)
  else:
    return (rotImg, rotRows)


def orient2(img, name = ""): # figure out if upside down or not

  (img, bigRows) = orient1(img)
  tolerance = 700

  data = list(img.getdata())
  #assert((0,0,0,255) in data)

  minAccRowSize = 5

  pixels = []
  w, h = img.size
  for i in range(h):
    pixels.append(data[i*w:(i+1)*w])

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
      pixel = pixels[curY][curX]
      diff = pixel[0] + pixel[1] + pixel[2]
      if diff < tolerance:
        nextRow += 1
      curX += 1

    posTop = curRow - prevRow
    posBot = curRow - nextRow
    if posTop > posBot and posTop > 0:
      rowTops[curY-1] = posTop

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

  orient = {}
  top = 0
  bottom = 0

  """
  final = Image.new("RGB", img.size)
  drawFin = ImageDraw.Draw(final)
  final.paste(img, (0,0))
  for (start,end) in bigRows:
    drawFin.line([(0,start), (w,start)], fill=(255,0,0))
    drawFin.line([(0,end), (w,end)], fill=(255,0,255))

  #for (start,end) in smallRows:
  #  drawFin.line([(0,start), (w,start)], fill=(0,255,0))
  #  drawFin.line([(0,end), (w,end)], fill=(0,0,255))
  final.save("wipRows" + name,"PNG")
  """

  assert( len(smallRows) == len(bigRows) )

  for c in range(len(smallRows)):
    if smallRows[c][1] - smallRows[c][0] < minAccRowSize:
      continue
    curY = bigRows[c][0]
    while curY < smallRows[c][0]:
      curX = 0
      while curX < w:
        if curY >= 0 and curY < len(pixels):
          pixel = pixels[curY][curX]
          diff = pixel[0] + pixel[1] + pixel[2]
          if diff < tolerance:
            top += 1
        curX += 1
      curY += 1

    curY = smallRows[c][1] + 1
    while curY <= bigRows[c][1]:
      curX = 0
      while curX < w:
        if curY >= 0 and curY < len(pixels):
          pixel = pixels[curY][curX]
          diff = pixel[0] + pixel[1] + pixel[2]
          if diff < tolerance:
            bottom += 1
        curX += 1
      curY += 1

  print top, bottom
  if top >= bottom:
    return img
  else:
    return imgRotate(img, 180) #img.rotate(180, Image.BICUBIC)

def simRBH(f, sx, sy, rand = False): # simulate prim with failure prob f
  states = []
  for r in range(sy):
    for c in range(sx):
      states.append((r,c))

  edgesX = []
  edgesY = []
  grid = {}
  rgrid = {}
  avail = set()
  nr = [0]
  nc = [1]
  #nr = [0,0,1,-1]
  #nc = [1,-1,0,0]
  whiteLeft = filter(lambda (x,y): y==0, states)
  whiteRight = filter(lambda (x,y): y==sy-1, states)
  start = random.choice(whiteLeft)
  states.remove(start)
  whiteLeft.remove(start)
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
      #favail = [(r,c) for (r,c) in avail if r >= -start[0] and c >= -start[1] and r < sy-start[0] and c < sx-start[1]]
      next = random.choice(list(avail))
      cr, cc = next

      rnr, rnc = (cr, cc -1)

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
      if v not in rgrid:
        avail.add(v)

    if nextState in whiteLeft:
      whiteLeft.remove(nextState)

    if nextState in whiteRight:
      whiteRight.remove(nextState)
      if len(whiteLeft) > 0:
        rowStart = random.choice(whiteLeft)
        whiteLeft.remove(rowStart)
        states.remove(rowStart)
        grid[rowStart] = (next[0]+1, 0)
        rgrid[(next[0]+1, 0)] = rowStart
        avail = set([(next[0]+1, 1)])
      elif len(states) > 0:
        rowStart = random.choice(states)
        states.remove(rowStart)
        grid[rowStart] = (next[0]+1, 0)
        rgrid[(next[0]+1, 0)] = rowStart
        avail = set([(next[0]+1, 1)])

  edgesX = []
  edgesY = []
  #print len(grid)
  for (state,loc) in grid.items():
    for i in range(len(nr)):
      v = (loc[0] + nr[i], loc[1] + nc[i])
      if v in rgrid:
        fr = rgrid[min(v,loc)]
        to = rgrid[max(v,loc)]
        if nr[i] != 0 and (fr,to) not in edgesY:
          edgesY.append((fr,to))
        elif nc[i] != 0 and (fr,to) not in edgesX:
          edgesX.append((fr,to))

  return edgesX, edgesY

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
  #nr = [0,0]
  #nc = [1,-1]
  nr = [0,0,1,-1]
  nc = [1,-1,0,0]
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
      if v not in rgrid:
        avail.add(v)

  edgesX = []
  edgesY = []
  #print len(grid)
  for (state,loc) in grid.items():
    for i in range(len(nr)):
      v = (loc[0] + nr[i], loc[1] + nc[i])
      if v in rgrid:
        fr = rgrid[min(v,loc)]
        to = rgrid[max(v,loc)]
        if nr[i] != 0 and (fr,to) not in edgesY:
          edgesY.append((fr,to))
        elif nc[i] != 0 and (fr,to) not in edgesX:
          edgesX.append((fr,to))

  return edgesX, edgesY

def simPrim1(f, sx, sy, rand = False): # simulate prim_correction with failure prob f
  states = []
  for r in range(sy):
    for c in range(sx):
      states.append((r,c))

  grid = {}
  rgrid = {}
  found = set()
  avail = set()
  #nr = [0,0]
  #nc = [1,-1]
  nr = [0,0,1,-1]
  nc = [1,-1,0,0]
  start = random.choice(states)
  found.add(start)
  grid[start] = (0,0)
  #print "start", start, "on (0, 0)"
  rgrid[(0,0)] = start
  cr, cc = (0,0)
  for i in range(len(nr)):
    avail.add((cr + nr[i], cc + nc[i]))

  while len(states) > len(found):

    navail = set()
    for a in grid.values():
      for i in range(len(nr)):
        v = (a[0] + nr[i], a[1] + nc[i])
        if v not in rgrid:
          navail.add(v)
    avail = navail

    #print avail
    nextState = None
    q = random.random()
    if q < f:
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
          #print len(states)
          if nstates == []:
            continue
          nextState = random.choice(nstates)
          #print nextState, next, neigh
          break

    else:
      #print q
      #print "non-random"
      favail = [(r,c) for (r,c) in avail if r >= -start[0] and c >= -start[1] and r < sy-start[0] and c < sx-start[1]]
      random.shuffle(favail)
      for next in favail:
        cr, cc = next

        fd = 0
        for i in range(len(nr)):
          v = (cr + nr[i], cc + nc[i])
          if v in rgrid:
            rnr, rnc = v
            fd += 1

        if fd == 0:
          continue
        ar, ac = (cr - rnr, cc - rnc)
        csr, csc = rgrid[(rnr,rnc)]
        corState = (csr + ar, csc + ac)
        if corState in states:      
          nextState = corState
          #print "determ", nextState, "on", next
        else:
          nextState = random.choice(states)
          #print "determ", corState, "not avail, so",nextState, "on", next
        break
    #print len(avail)
    avail.remove(next)
    if nextState == None:
      nextState = random.choice(states)

    if nextState not in found:
      found.add(nextState)
    else:
      avail.add(grid[nextState])
      del rgrid[grid[nextState]]

    grid[nextState] = next
    rgrid[next] = nextState

  edgesX = []
  edgesY = []
  #print len(grid)
  for (state,loc) in grid.items():
    for i in range(len(nr)):
      v = (loc[0] + nr[i], loc[1] + nc[i])
      if v in rgrid:
        fr = rgrid[min(v,loc)]
        to = rgrid[max(v,loc)]
        if nr[i] != 0 and (fr,to) not in edgesY:
          edgesY.append((fr,to))
        elif nc[i] != 0 and (fr,to) not in edgesX:
          edgesX.append((fr,to))

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


def getDecisionTree(ims):
  context = []
  extentSize = 6
  for f in range(-extentSize,extentSize+1):
    for s in range(-extentSize,0):
      if not (f==0 and s==0):
        context.append((f,s)) 

  features = {}
  categs = []
  for c in context:
    features[c] = []

  for im in ims:
    w,h = im.size
    ld = list(im.getdata())
    
    bd = []
    for r in range(h):
      bd.append([])
      for c in range(w):
        bd[r].append(ld[r*w + c])

    div = 255.0
    for r in range(extentSize, h -extentSize):
      for c in range(extentSize, w):
        vr = bd[r][c]/div
        categs.append(vr)
        for i in context:
          nr = r+i[0]
          nc = c+i[1]
          features[i].append(bd[nr][nc]/div)

  print getFeature(categs, features, 5)

def getFeature(categs, features, maxDepth, depth=1, pReach = 1.0):
  minEnt = 1.0
  bestFeature = None
  bestData = None

  for (k,v) in features.items():
    count = 0.0
    blacks = 0.0
    predB = 0.0
    predW = 0.0    
    for i in range(len(v)):
      ct = categs[i]
      count += 1
      if v[i] == 0:
        blacks += 1
        if ct == 0:
          predB += 1
      else:
        if ct == 1:
          predW += 1

    probB = blacks/count
    probW = 1 - probB
    whites = count - blacks

    ent = 0
    if probB > 0:
      ent += probB * getH2(predB / blacks) 
    if probW > 0:
      ent += probW * getH2(predW / whites)

    if ent < minEnt:
      minEnt = ent
      bestFeature = k
      bestData = probB, probW, predB / blacks, predW / whites

  catPredB = "B"
  predB = bestData[2]
  if predB < 0.5:
    catPredB = "W"
    predB = 1 - predB

  catPredW = "W"
  predW = bestData[3]
  if predW < 0.5:
    catPredW = "W"
    predW = 1 - predW

  probCorr = bestData[0] * predB + bestData[1] * predW

  print depth, pReach, bestData, bestFeature, minEnt, catPredB, catPredW, probCorr

  if depth >= maxDepth:
    return probCorr * pReach

  nCatB = []
  nCatW = []
  setFeatB = set()
  nFeatB = {}
  nFeatW = {}
  for c in features.keys():
    if c == bestFeature:
      continue
    nFeatB[c] = []
    nFeatW[c] = []

  bv = features[bestFeature]
  for i in range(len(bv)):
    ct = categs[i]
    if bv[i] == 0:
      nCatB.append(ct)
      setFeatB.add(i)
    else:
      nCatW.append(ct)

  for (k,v) in features.items():
    if k == bestFeature:
      continue
    for i in range(len(v)):
      if i in setFeatB:
        nFeatB[k].append(v[i])
      else:
        nFeatW[k].append(v[i])

  pcB = getFeature(nCatB, nFeatB, maxDepth, depth+1, pReach*bestData[0])
  pcW = getFeature(nCatW, nFeatW, maxDepth, depth+1, pReach*bestData[1])

  return pcB + pcW

def getContextEffects(ims):
  context = []
  extentSize = 6
  for f in range(-extentSize,extentSize+1):
    for s in range(-extentSize,0):
      context.append((f,s)) 
  pixCount = {}
  totCount = {}
  pixPred = {}
  for c in context:
    totCount[c] = 0.0
    pixCount[c] = 0.0
    pixPred[c] = [0.0,0.0]

  for im in ims:
    w,h = im.size
    ld = list(im.getdata())
    
    bd = []
    for r in range(h):
      bd.append([])
      for c in range(w):
        bd[r].append(ld[r*w + c])

    div = 255.0
    for r in range(h):
      for c in range(w):
        vr = bd[r][c]/div
        for i in context:
          nr = r+i[0]
          nc = c+i[1]
          if nr >= 0 and nr < h and nc >= 0 and nc < w:
            totCount[i] += 1
            if bd[nr][nc] == 0:
              pixCount[i] += 1
              if vr == 0:
                pixPred[i][0] += 1
            else:
              if vr == 0:
                pixPred[i][1] += 1

  #print pixCount, totCount, pixPred
  rez = {}
  sol = None
  minH2 = 1.0
  for (k,v) in pixPred.items():
    count = totCount[k]
    blackCount = pixCount[k]
    whiteCount = count - pixCount[k]
    probBlack = blackCount / count
    probWhite = 1 - probBlack
    h2 = probBlack * getH2(v[0]/blackCount) + probWhite * getH2(v[1]/whiteCount)
    rez[k] = (v[0]/blackCount, v[1]/whiteCount, h2)
    #print k, v[0]/blackCount, v[1]/whiteCount, h2)

  print '\n'.join(map(str, sorted(rez.items(), key=lambda x: x[1][2])))
  #print sol

def getPrediction7(ims):
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
      for c in range(2,w-1):
        nl = (bd[r-1][c]/div,bd[r][c]/div,bd[r+1][c]/div,bd[r-1][c-1]/div, bd[r-1][c+1]/div,bd[r][c+1]/div,bd[r+1][c+1]/div)
        vl = bd[r][c-1]/div
        try:
          predXl[nl][vl] += 1
        except:
          predXl[nl] = {vl : 2.0, 1-vl : 1.0}

        nr = (bd[r-1][c-1]/div,bd[r][c-1]/div,bd[r+1][c-1]/div,bd[r-1][c]/div, bd[r-1][c-2]/div,bd[r][c-2]/div,bd[r+1][c-2]/div)
        vr = bd[r][c]/div
        try:
          predXr[nr][vr] += 1
        except:
          predXr[nr] = {vr : 2.0, 1-vr : 1.0}

    for r in range(2,h-1):
      for c in range(1,w-1):
        nl = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div,bd[r-1][c-1]/div, bd[r+1][c-1]/div,bd[r+1][c]/div,bd[r+1][c+1]/div)
        vl = bd[r-1][c]/div
        try:
          predYl[nl][vl] += 1
        except:
          predYl[nl] = {vl : 2.0, 1-vl : 1.0}

        nr = (bd[r-1][c-1]/div,bd[r-1][c]/div,bd[r-1][c+1]/div,bd[r][c-1]/div, bd[r-2][c-1]/div,bd[r-2][c]/div,bd[r-2][c+1]/div)
        vr = bd[r][c]/div
        try:
          predYr[nr][vr] += 1
        except:
          predYr[nr] = {vr : 2.0, 1-vr : 1.0}

  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              for i7 in [0,1]:
                n = tuple(map(float,(i1,i2,i3,i4,i5,i6,i7)))
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
                print n, predXl[n][0.0] + predXl[n][1.0]
                predXl[n] = normalize(predXl[n])
                predYl[n] = normalize(predYl[n])
                predXr[n] = normalize(predXr[n])
                predYr[n] = normalize(predYr[n])

  return prior, predXl, predYl, predXr, predYr

def getPrediction4(ims):
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
        nl = (bd[r-1][c]/div,bd[r][c]/div,bd[r+1][c]/div,bd[r-1][c-1]/div)#, bd[r-1][c+1]/div,bd[r][c+1]/div,bd[r+1][c+1]/div)
        vl = bd[r][c-1]/div
        try:
          predXl[nl][vl] += 1
        except:
          predXl[nl] = {vl : 2.0, 1-vl : 1.0}

        nr = (bd[r-1][c-1]/div,bd[r][c-1]/div,bd[r+1][c-1]/div,bd[r-1][c]/div)#, bd[r-1][c-2]/div,bd[r][c-2]/div,bd[r+1][c-2]/div)
        vr = bd[r][c]/div
        try:
          predXr[nr][vr] += 1
        except:
          predXr[nr] = {vr : 2.0, 1-vr : 1.0}

    for r in range(1,h):
      for c in range(1,w-1):
        nl = (bd[r][c-1]/div,bd[r][c]/div,bd[r][c+1]/div,bd[r-1][c-1]/div)#, bd[r+1][c-1]/div,bd[r+1][c]/div,bd[r+1][c+1]/div)
        vl = bd[r-1][c]/div
        try:
          predYl[nl][vl] += 1
        except:
          predYl[nl] = {vl : 2.0, 1-vl : 1.0}

        nr = (bd[r-1][c-1]/div,bd[r-1][c]/div,bd[r-1][c+1]/div,bd[r][c-1]/div)#, bd[r-2][c-1]/div,bd[r-2][c]/div,bd[r-2][c+1]/div)
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
          #print n, predXl[n][0.0] + predXl[n][1.0]
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

def getH2(prob):
  if prob == 1 or prob == 0:
    return 0
  rp = 1.0 - prob
  return -prob*math.log(prob,2) - rp*math.log(rp,2)

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
    prob = {}
    im = Image.open("SampleDocs/text31.jpg").convert("1")
    #getContextEffects([im])
    
    #dt = DTree([im],5)
    rawDataV = """
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
"""

    rawDataH = """
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
"""
    #dataH = map(lambda x: map(int,x.strip().split(' ')), rawDataH.strip().split('\n'))
    #dataV = map(lambda x: map(int,x.strip().split(' ')), rawDataV.strip().split('\n'))
    
    #print dt.getProb(dataV, 3, 0, "V"), dt.getProb(dataV, 3, 255, "V"), dt.getProb(dataH, 3, 0, "H"), dt.getProb(dataH, 3, 255, "H")
    #assert(False)

    prior, prxl, pryl, prxr, pryr = getPrediction4([im])
    #print prior
    #print prxl

    for i in range(2,11):#[10,15,25,35]:
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
      
      page = pages.ImagePage(sx, sy, im)
      #page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
      page.setCost("blackGaus")



      #page.setCost("decisionTree", dt = dt)
      
      #page.setCost("blackGaus")
      #page.output(page.states[(1,1)])
      #print cost.indivPicker(page.costType,(1,1), (-42,-42), "x",  page, False)
      #g1DMat = search.picker("g1D", page)
      #page.heapify()
      ###(pos, edges) = search.picker("kruskal", page)
      #page.heapify()
      #(kpos, kedges) = search.picker("kruskal", page)
      #print g1DMat
      #print edges
      #print pos
      #print convertMatPos(g1DMat)
      ###correct += sum(page.totalCost)
      #g1D += sum(page.calcCostMat(g1DMat))
      ###prm += sum(page.calcCostList(pos))
      ###intCorrect += sum(page.totalInternalCost)
      #intG1D += sum(page.calcCostMat(g1DMat, True))

      ###intPrm += sum(page.calcCostList(pos, True))

      #evalCost = cost.evaluateProb(page, sx, sy)
      evalCost = cost.evaluateCost(page, sx, sy)
      """
      ps = evalCost[2]
      for (k,(v,n,q)) in ps.items():
        try:
          prob[k] = (prob[k][0] + v, prob[k][1] + n, prob[k][2]+q)
        except:
          prob[k] = (v,n,q)

      #print sorted([(x,q/z,y/z,z) for (x,(y,z,q)) in ps.items()])
      sys.stdout.flush()
      costEval.append(evalCost[0])
      """
      print i*i, evalCost[0]
      costEval.append(evalCost[0])

      ###print intCorrect/reps, intPrm/reps#, intG1D/reps
      ###print correct/reps, prm/reps#, g1D/reps
      #print "cost eval", evalCost
      ###edgeP = page.calcCorrectEdges(pos)
      #edgeK = page.calcCorrectEdges(kpos)
      #edgeG = page.calcCorrectEdges(convertMatPos(g1DMat))
      ###edgeC = 1.0
      ###print "edges", edgeP#, edgeK, edgeG
      #print sorted(page.calcCostList(convertMatPos(g1DMat), False)) == sorted(page.calcCostMat(g1DMat, False))
      ###c.append(correct)
      ###p.append(prm)
      #g.append(g1D)
      ###ec.append(edgeC)
      ###ep.append(edgeP)
      #eg.append(edgeG)
      #ek.append(edgeK)
      ###ic.append(intCorrect)
      ###ip.append(intPrm)
      #ig.append(intG1D)
      ###y.append(i*i)
      #print sum(page.calcCostList({(0, 1): (0, 1), (1, 0): (1, 0), (0, 0): (0, 0), (1, 1): (1, 1)}))

    #print ' '.join(map(str,c))
    #print ' '.join(map(str,ic))
    #print ' '.join(map(str,p))
    #print ' '.join(map(str,ip))
    #print ' '.join(map(str,ep))

    print costEval
    """
    print "total"
    print sorted([(x,q/z,y/z,z) for (x,(y,z,q)) in prob.items()])
    
    i20 = [(0, 0.0091290199977457874, 0.0068998597138859733, 286), (1, 0.092875041027483327, 0.066666666666666666, 30), (2, 0.20176894214720276, 0.083333333333333329, 42), (3, 0.30044528043338581, 0.2011494252873563, 29), (4, 0.39621885825235331, 0.19318181818181818, 44), (5, 0.4962001060077475, 0.43518518518518517, 54), (6, 0.59526841334815284, 0.18518518518518517, 27), (7, 0.70110502957891263, 0.30188679245283018, 53), (8, 0.80761362174707341, 0.57692307692307687, 26), (9, 0.90416987434502882, 0.58695652173913049, 46), (10, 0.99045721751689209, 0.84552845528455289, 123)]


    i15 = [(0, 0.013073051832917114, 0.015012214045887347, 123), (1, 0.082391694555403835, 0.22222222222222221, 6), (2, 0.19853881242630023, 0.15972222222222221, 24), (3, 0.30610217401542761, 0.30952380952380948, 14), (4, 0.40082441591543333, 0.22727272727272727, 22), (5, 0.50312162443339137, 0.35483870967741937, 31), (6, 0.60729359843637487, 0.52173913043478259, 23), (7, 0.70220134833037362, 0.375, 16), (8, 0.80751561857713794, 0.55555555555555558, 27), (9, 0.90423706913703805, 0.58333333333333337, 24), (10, 0.99330265415059882, 0.84545454545454546, 110)]

    i25 = [(0, 0.0065074182268112946, 0.0098764975360062564, 456), (1, 0.097282393877329995, 0.085826620636747217, 79), (2, 0.20443546027653856, 0.1134831460674157, 89), (3, 0.29711203602975828, 0.22286821705426357, 86), (4, 0.39354827331547881, 0.18604651162790697, 86), (5, 0.49714061259646991, 0.29054054054054052, 74), (6, 0.59607211095047263, 0.28888888888888886, 45), (7, 0.70224646123796075, 0.4838709677419355, 31), (8, 0.80420865703465916, 0.43333333333333335, 30), (9, 0.90498060789312651, 0.39344262295081966, 61), (10, 0.9875027577745592, 0.86503067484662577, 163)]

    i35 = [(0, 0.0049548121476102333, 0.0043816311259321679, 1058), (1, 0.10178276332411157, 0.068770372360450666, 383), (2, 0.19033366043199831, 0.1202294685990338, 276), (3, 0.30013198327547702, 0.16129032258064516, 155), (4, 0.40233988394593101, 0.19047619047619047, 105), (5, 0.49616325472001949, 0.28488372093023256, 86), (6, 0.60431272339467534, 0.31578947368421051, 57), (7, 0.69794424446217418, 0.22222222222222221, 63), (8, 0.80592706421796689, 0.59259259259259256, 54), (9, 0.90944283777256696, 0.68493150684931503, 73), (10, 0.98215014964015013, 0.91428571428571426, 70)]


    x20,l20 = ([q[0]/10.0 for q in i20], [q[2] for q in i20])
    x15,l15 = ([q[0]/10.0 for q in i15], [q[2] for q in i15])
    x25,l25 = ([q[0]/10.0 for q in i25], [q[2] for q in i25])
    x35,l35 = ([q[0]/10.0 for q in i35], [q[2] for q in i35])
    #xTotal,lTotal = ([q[0] for q in iTotal], [q[2] for q in iTotal])

    plt.plot(x35,l35, 'r-*', x25,l25, 'g-H', x15,l15, 'b-d', x20,l20, 'm-+')#, x5,l5, 'c')#,xTotal,lTotal, 'k')
    plt.annotate("35*35", (x35[3],l35[3]), xytext = (0.2, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("25*25", (x25[6],l25[6]), xytext = (0.2, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("15*15", (x15[3],l15[3]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("20*20", (x20[5],l20[5]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("25*25", (x25[13],l25[13]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Total Sum", (xTotal[13],lTotal[13]), xytext = (0.2, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.show()
    """

  elif "4" in arg:
    ys = []
    im = Image.open("SampleDocs/text31.jpg").convert("1")

    #w, h = im.size
    #im = im.resize((w/2, h/2), Image.ANTIALIAS)
    #prior, prxl, pryl, prxr, pryr = getPrediction([im])
    errors = {}
    unknowns = {}
    for j in [1,10,50,100]:#range(30,91,10):
      errors[j] = []
      unknowns[j] = []
      for i in range(10,201,20):
        sx = i
        sy = j

        page = pages.ImagePage(sx, sy, im)

        #boxes = page.getBoxes()
        #bigRows, smallRows = page.getRows1()
        page.calcPiecesRows()
        
        white, unknown, upside = page.calcOrients()
        errors[j].append(float(upside) / (i*j))
        unknowns[j].append(float(unknown) / (i*j))
        print j, i, white, unknown, upside, float(white) / (i*j), float(unknown) / (i*j), float(upside) / (i*j), float(unknown + upside) / (i*j)
      #print smallRows, len(smallRows)
      #final = Image.new("RGB", im.size)
      #drawFin = ImageDraw.Draw(final)
      #final.paste(im, (0,0))
      #if boxes != None:
      #  for (start,end) in boxes:
      #    drawFin.rectangle([(start[1],start[0]), (end[1],end[0])], outline=(255,0,0))

      #if bigRows != None:
      #  for (start,end) in bigRows:
      #    drawFin.line([(0,start), (im.size[0],start)], fill=(255,0,0))
      #    drawFin.line([(0,end), (im.size[0],end)], fill=(255,0,255))

      #if smallRows != None:
      #  for (start,end) in smallRows:
      #    drawFin.line([(0,start), (im.size[0],start)], fill=(0,255,0))
      #    drawFin.line([(0,end), (im.size[0],end)], fill=(0,0,255))
      #final.save("ghhghh","JPEG")
      #break
      #print page.whites

      #print verifCost.checkAll(page)
      #page.setCost("gaus")
      #page.setCost("blackGaus")
      
      ###page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
      
      #g1DMat = search.picker("g1D", page)
      #page.heapify()

      ###(pos, edges) = search.picker("prim", page)
      ###page.vizPos(pos, None, False, True)

      #break
      #page.heapify()
      #(kPos, kEdges) = search.picker("kruskal", page, True)
      #print edges
      #print sorted(pos.items(), key=lambda x: x[1])
      #print g1DMat
      #print sum(page.totalInternalCost), sum(page.calcCostList(kPos, True)), sum(page.calcCostList(pos, True)), sum(page.calcCostMat(g1DMat, True))
      #print sum(page.totalCost),sum(page.calcCostList(kPos)), sum(page.calcCostList(pos)), sum(page.calcCostMat(g1DMat))
      ###cc = cost.evaluateCost(page, sx, sy)
      ###print i, cc#,page.calcCorrectEdges(kPos, True), page.calcCorrectEdges(pos), page.calcCorrectEdges(convertMatPos(g1DMat))
      ###ys.append(cc[0])
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
    print errors
    print unknowns
  elif "5" in arg:
    #px, py = getPercents([Image.open("SampleDocs/text3.jpg").convert("1")])
    #print "orig px", '\n'.join(map(str,sorted(px.items(), key=lambda x : x[1]))), '\n======'
    #print "orig py",  '\n'.join(map(str,sorted(py.items(), key=lambda x : x[1]))), '\n======'
    im = Image.open("SampleDocs/p01.png").convert("1")
    prior, prxl, pryl, prxr, pryr = getPrediction4([im])

    #print '\n'.join(map(str, prxl.items()))
    #print prior
    #print "orig prxr", '\n'.join(map(str,sorted(prxr.items(), key=lambda x : x[1][0.0]))), '\n======'
    #print "orig prxl",  '\n'.join(map(str,sorted(prxl.items(), key=lambda x : x[1][0.0]))), '\n======'
    ns = []
    xs = []
    ys = []
    ec = []
    for i in range(2,10):
      start = time.clock()
      sx = i
      sy = i
      #print "ff"
      page = pages.ImagePage(sx, sy, im)
      #rr = page.confEdges(4)
      #print i, rr
      #xs.append(rr)
      #npx, npy = getPercents(page.states.values())
      #print "piece px", sorted(px, key=lambda x : x[1]), '\n======'
      #print "piece py", sorted(py, key=lambda x : x[1]), '\n======'
      #print "qq"
      page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
      #page.setCost("nn")

      #evalCost = cost.evaluateProb(page, sx, sy)
      #print i, evalCost[0]
      #ec.append(evalCost[0])

      pPos, pEdges = search.picker("prim", page)
      #rez = search.picker("g1D", page)
      #pPos = convertMatPos(rez)

      gp = page.calcGroups(pPos)
      groupP = (len(gp),sorted(gp, reverse = True) )
      #page.heapify()
      #(kPos, kEdges) = search.picker("kruskalMulti", page)
      #gk = page.calcGroups(kPos)
      #groupK = (len(gk),sorted(gk, reverse = True) )
      corrp = page.calcCorrectEdges(pPos)
      corrk = 0 #page.calcCorrectEdges(kPos)
      ns.append(i*i)
      xs.append(corrp)
      ys.append(corrk)
      sys.stderr.write(str(i) + " " + str(corrp) + " " + str(groupP[0]) + " " + str(time.clock() - start) + "\n")
      #break      
      #print groupP
      #print groupK

    print ec
    #print "prim", xs
    #print "kruskal", ys
    plt.plot(ns,xs, 'r-', ns,ys, 'g-')
    
    #plt.xlabel("Number of shreds")
    #plt.ylabel("Proportion of shreds with 4 or less black pixels on each edge")
    plt.show()
 
  elif "6" in arg:
    #ex1, ey1 = genEdges(5,5)
    #for i in range(1):
      #ex2, ey2 = simKruskal(0.9,20,20)
      #print ex1, ey1
      #print ex2, ey2
      #print simError(ex1,ey1,ex2,ey2)
    #assert False
    sx = 15
    sy = 15
    count = 50
    xs = []
    ys = []
    ts = []
    ns = []

    ex12, ey12 = genEdges(5,5)
    print len(ex12), len(ey12)
    #ex13, ey13 = genEdges(4,256)
    #ex14, ey14 = genEdges(32,32)
    for f in np.arange(0.0, 1.01, 0.01):
      sumXp = 0.0
      sumYp = 0.0
      sumTp = 0.0
      sumXk = 0.0
      sumYk = 0.0
      sumTk = 0.0
      sumXq = 0.0
      sumYq = 0.0
      sumTq = 0.0
      for i in range(count):
        ex2, ey2 = simRBH(f,5,5)
        #ex3, ey3 = simPrim1(f,5,5)
        #print len(ex3), len(ey3)
        #ex4, ey4 = simPrim(f,32,32)
        #print ex1, ey1
        #print ex2, ey2
        #xk,yk,tk = simError(ex12,ey12,ex2,ey2)
        #sumXk += xk
        #sumYk += yk
        #sumTk += tk
        xp,yp,tp = simError(ex12,ey12,ex2,ey2)
        sumXp += xp
        sumYp += yp
        sumTp += tp
        #xq,yq,tq = simError(ex14,ey14,ex4,ey4)
        #sumXq += xq
        #sumYq += yq
        #sumTq += tq

      sumXp /= count
      sumYp /= count
      sumTp /= count

      #sumXk /= count
      #sumYk /= count
      #sumTk /= count

      #sumXq /= count
      #sumYq /= count
      #sumTq /= count

      print "Prim", f, sumXp, sumYp, sumTp
      #print "Prim1", f, sumXk, sumYk, sumTk
      xs.append(sumTp)
      #ys.append(sumTk)
      #ts.append(sumTq)
      ns.append(f)

    #print ts
    #print ys
    print xs
    plt.figure(1)
    plt.plot(ns,xs, 'r-', ns, [1-x for x in ns], 'b-')
    plt.show()
  elif "7" in arg:
    with open("9995Raw",'r') as f:
      s9995 = f.read() 
    with open("995Raw",'r') as f:
      s995 = f.read()
    with open("95Raw",'r') as f:
      s95 = f.read()
    
    """
    se9995 = map(lambda x: x[1:].split('['), re.findall('g[0-9. ,\[\]\(\)]*\)\]\)',filter(lambda x: x != '\n', s9995)))
    red9995 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[0]))[-1], se9995)
    per9995 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[1]))[::3], se9995)
    size9995 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[1]))[1::3], se9995)
    totSize9995 = map(lambda x: sum(map(float,x)),size9995)
    err9995 = [sum([float(per9995[i][j]) * float(size9995[i][j]) for j in range(len(per9995[i]))]) / float(totSize9995[i]) for i in range(len(totSize9995))]
    
    se995 = map(lambda x: x[1:].split('['), re.findall('g[0-9. ,\[\]\(\)]*\)\]\)',filter(lambda x: x != '\n', s995)))
    red995 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[0]))[-1], se995)
    per995 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[1]))[::3], se995)
    size995 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[1]))[1::3], se995)
    totSize995 = map(lambda x: sum(map(float,x)),size995)
    err995 = [sum([float(per995[i][j]) * float(size995[i][j]) for j in range(len(per995[i]))]) / float(totSize995[i]) for i in range(len(totSize995))]
    
    se95 = map(lambda x: x[1:].split('['), re.findall('g[0-9. ,\[\]\(\)]*\)\]\)',filter(lambda x: x != '\n', s95)))
    red95 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[0]))[-1], se95)
    per95 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[1]))[::3], se95)
    size95 = map(lambda x: filter(lambda y : len(y) > 0, re.findall('[0-9.]*',x[1]))[1::3], se95)
    totSize95 = map(lambda x: sum(map(float,x)),size95)
    err95 = [sum([float(per95[i][j]) * float(size95[i][j]) for j in range(len(per95[i]))]) / float(totSize95[i]) for i in range(len(totSize95))]
    print totSize95
    xs = [i*i for i in range(2,16)]
    #plt.plot(xs,err9995, 'g-H', linewidth=3, markersize=12)
    #plt.plot(xs, err995, 'r-*', linewidth=3, markersize=12)
    #plt.plot(xs, err95, 'b-d', linewidth=3, markersize=12)
    plt.plot(xs,red9995, 'g-H', linewidth=3, markersize=12)
    plt.plot(xs, red995, 'r-*', linewidth=3, markersize=12)
    plt.plot(xs, red95, 'b-d', linewidth=3, markersize=12)
    
    plt.xlabel("Number of shreds", size=30)
    plt.ylabel("Search space reduction", size=30)
    #plt.ylabel("Proportion correct edges")#, size=20)
    plt.tick_params(axis='both', labelsize=30)
    plt.annotate("95%", (xs[9],red95[9]), fontsize=25, xytext = (20, -40),  textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("99.5%", (xs[11],red995[11]), fontsize=25, xytext = (60, 30),  textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("99.95%", (xs[8],red9995[8]), fontsize=25, xytext = (30, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    #plt.annotate("95%", (xs[11],err95[11]), xytext = (20, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("99.5%", (xs[13],err995[13]), xytext = (30, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("99.95%", (xs[10],err9995[10]), xytext = (35, -25), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

    a = plt.gca()
    a.set_ylim([0,1.0])
    plt.subplots_adjust(left=0.19, right=0.94, top=0.94, bottom=0.15)
    plt.show()
    """
    """
    bg =[0.80000000000000004, 0.77499999999999991, 0.51715686274509798, 0.50961538461538469, 0.37567567567567578, 0.39785714285714285, 0.37994505494505526, 0.51058604336043412, 0.31468646864686495, 0.33885991058122245, 0.31266980146290535, 0.26257541478129698, 0.23832766218553006, 0.24430046354825141, 0.23577739948119428, 0.23483772819472648, 0.27312468577174348, 0.25027866627895617, 0.17653782211138913]

    bgS1 = [0.80000000000000004, 0.77499999999999991, 0.60049019607843135, 0.65961538461538471, 0.56734234234234238, 0.4330952380952382, 0.38901098901098929, 0.46553184281842852, 0.32307480748074835, 0.37067809239940436, 0.34097439916405475, 0.31389517345399665, 0.21197774306911341, 0.24552254530130665, 0.20417207090358941, 0.20168610547667373, 0.21838612368024038, 0.1973630361969998, 0.13505435971474938]

    bgD15=[0.80000000000000004, 0.76666666666666661, 0.52696078431372539, 0.48461538461538467, 0.32612612612612624, 0.20952380952380942, 0.27740384615384611, 0.44478319783197873, 0.25962596259625992, 0.2038002980625935, 0.21833855799373086, 0.16332956259426917, 0.22986277681709147, 0.17363042562157646, 0.15585278858625259, 0.16710826572008147, 0.13565191888721254, 0.20285962758338499, 0.10166633853961705]

    bgN01 = [1.0, 0.75, 0.50245098039215685, 0.46346153846153848, 0.38378378378378381, 0.40547619047619049, 0.30467032967032964, 0.4697662601626017, 0.25948844884488448, 0.3283904619970196, 0.28132183908045999, 0.23305052790346911, 0.20804745170227387, 0.21873156342182906, 0.19149238002594085, 0.2036806964164978, 0.23487347075582299, 0.20526396562308166, 0.14092236513978296]

    g = [0.58333333333333337, 0.78571428571428559, 0.54930555555555549, 0.49107142857142866, 0.50151282051282053, 0.48057133838383848, 0.48014133683776572, 0.5629844114219118, 0.38519148353408322, 0.42024187371275878, 0.37804269339450219, 0.40595504806416166, 0.3169970337368318, 0.31459838764519366, 0.28409745104446577, 0.2986933407921355, 0.33914342133293579, 0.33097146277829392, 0.24350653476865139]

    gS1 = [0.083333333333333329, 0.61111111111111116, 0.52499999999999991, 0.50416666666666665, 0.3653637566137567, 0.32734772069747442, 0.37196575126262627, 0.40771842486781512, 0.29894866097940986, 0.29025678574191238, 0.29772215762090676, 0.24823862859012599, 0.18749510298392447, 0.19769368819849203, 0.18726326742244212, 0.16572105944047236, 0.18408309821080471, 0.18387389844794888, 0.13633024400463054]

    gD15 = [0.58333333333333337, 0.78571428571428559, 0.32959401709401698, 0.41250000000000009, 0.4634081196581199, 0.25885996097689645, 0.30090702947845799, 0.46614587986539213, 0.33605977462682551, 0.34502226153561127, 0.26135871007497691, 0.18248812657907118, 0.30656756826867315, 0.24194728561680573, 0.173538960832294, 0.24199637974912272, 0.17105737346303648, 0.25830401711607281, 0.12840981415346278]

    gN01 = [0.25, 0.41666666666666669, 0.25, 0.28749999999999998, 0.24166666666666667, 0.18849206349206349, 0.3125, 0.37754629629629627, 0.21481481481481485, 0.23322510822510822, 0.23683425160697888, 0.16312321937321941, 0.17664581795016579, 0.1743640350877193, 0.17069196428571426, 0.19848442192192192, 0.22230846042120553, 0.18013356363579194, 0.12641072607153861]


    ceP01 = [0.83333333333333337, 0.78571428571428559, 0.73333333333333339, 0.56250000000000011, 0.5848461538461539, 0.55640106421356439, 0.53290107709750612, 0.66639876327376368, 0.47658037242297213, 0.49462497488831275, 0.47456038732868699, 0.4577243257949779, 0.38022948376897703, 0.39844405316105758, 0.33730559887765532, 0.35500473869091898, 0.4049667472915936, 0.37398131139288998, 0.27935091212040064]

    ceP01N01 = [0.5, 0.5, 0.25, 0.27500000000000002, 0.31666666666666665, 0.23809523809523808, 0.22321428571428573, 0.34722222222222221, 0.19444444444444445, 0.18636363636363637, 0.17803030303030304, 0.17307692307692307, 0.14285714285714285, 0.15238095238095239, 0.12916666666666668, 0.13051470588235295, 0.18300653594771241, 0.14342105263157895, 0.099404761904761912]

    ceP01S1 = [0.83333333333333337, 0.86111111111111105, 0.56666666666666654, 0.59166666666666656, 0.56142857142857139, 0.4947660098522168, 0.47097368777056275, 0.53677358521108554, 0.37162804462370574, 0.37649637992669349, 0.36716328569738321, 0.34871035475593737, 0.24795041163949696, 0.28920157434418264, 0.26236539160544864, 0.20410522782676188, 0.26553916204266487, 0.24747212443233477, 0.17352132229138117]

    ceP01D15 = [0.58333333333333337, 0.78571428571428559, 0.48611111111111116, 0.48869047619047629, 0.4776056505223174, 0.31670026881720426, 0.37307964852607728, 0.48932709679660902, 0.38207829314534397, 0.40164902449149009, 0.26878340862036515, 0.20215104135692849, 0.31426862492806312, 0.23318901237499315, 0.19409114173815625, 0.2642352817804971, 0.18245958228412834, 0.28917584550543329, 0.13892766180777641]

    c7 = [0.83333333333333337, 0.78571428571428559, 0.73333333333333339, 0.54345238095238102, 0.54317948717948716, 0.50585091991342002, 0.49718679138322031, 0.60158840002590042, 0.42658037242297203, 0.50371759795152415, 0.45767962316531413, 0.48016022323087537, 0.37438263701303937, 0.38100898822599272, 0.32888613953557638, 0.35814120853010351, 0.38771836054023096, 0.38415674448394904, 0.30750685781714499]

    c7N01 = []
    c7S1 = []
    c7D15 = []
    minInd = 0

    diffN01 = [y/x for (x,y) in zip(g,gN01)]
    diffS1 = [y/x for (x,y) in zip(g,gS1)]
    diffD15 = [y/x for (x,y) in zip(g,gD15)]

    xs = [i*i for i in range(2,21)][minInd:]

    #plt.plot(xs[minInd:],diffN01[minInd:], 'g-H', xs[minInd:], diffD15[minInd:], 'r-*', xs[minInd:], diffS1[minInd:], 'b-d')

    #plt.plot(xs,ceP01, 'g-H', xs, ceP01D15, 'r-*', xs, ceP01N01, 'b-d', xs, ceP01S1, 'm->')
    
    p1 = [ceP01[minInd:], ceP01D15[minInd:], ceP01S1[minInd:], ceP01N01[minInd:]]
    p2 = [bg[minInd:], bgD15[minInd:], bgS1[minInd:], bgN01[minInd:]]
    p1Text = [(70, 15),(120, 25),(120, 35),(70, -55)]
    p2Text = [(40, -30),(120, -50),(90, -60),(80, 45)]
    p1Ind = [11,4,6,3]
    p2Ind = [8,3,4,10]
    ind = 0
    
    plt.plot(xs,p1[ind], 'g-^')#, linewidth=3, markersize=15)
    plt.plot(xs, p2[ind], 'r-*')#, linewidth=3, markersize=20)

    plt.annotate("ProbScore", (xs[p1Ind[ind]],p1[ind][p1Ind[ind]]), p1Text[ind], fontsize=20, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=1))
    plt.annotate("GaussCost", (xs[p2Ind[ind]],p2[ind][p2Ind[ind]]), p2Text[ind], fontsize=20, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=1))
    
    #plt.annotate("Original Image", (xs[10],ceP01[10]), xytext = (50, 15), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Downsampled", (xs[12],ceP01D15[12]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Flipped pixels", (xs[8],ceP01N01[8]), xytext = (40, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Shuffled pixels", (xs[16],ceP01S1[16]), xytext = (50, 15), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

    #plt.annotate("Downsampled", (xs[12],diffD15[12]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Flipped pixels", (xs[7],diffN01[7]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Shuffled pixels", (xs[14],diffS1[14]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

    
    plt.xlabel("Number of shreds", size=20)
    plt.ylabel("Proportion correct edges", size=20)
    #plt.xticks([100,200,300,400])
    #plt.yticks([0.1,0.2,0.3,0.4,0.5])
    plt.tick_params(axis='both', labelsize=20)
    a = plt.gca()
    a.set_ylim([0,1])
    #a.set_xlim([100,400])
    #plt.subplots_adjust(left=0.16, right=0.94, top=0.94, bottom=0.15)
    plt.show()
    """
    
    a = [1.0, 0.95924999999999994, 0.86175000000000024, 0.84450000000000003, 0.78900000000000015, 0.71624999999999983, 0.71700000000000019, 0.70824999999999994, 0.64249999999999996, 0.63174999999999992, 0.55525000000000002, 0.56450000000000033, 0.55025000000000002, 0.49775000000000008, 0.51724999999999999, 0.47599999999999992, 0.48950000000000016, 0.45925000000000005, 0.42825000000000008, 0.44024999999999997, 0.39824999999999994, 0.39075000000000015, 0.38525000000000004, 0.35549999999999998, 0.37799999999999989, 0.33949999999999997, 0.34899999999999998, 0.30175000000000018, 0.3040000000000001, 0.30774999999999997, 0.29999999999999993, 0.28549999999999992, 0.29249999999999993, 0.25200000000000006, 0.30374999999999991, 0.25999999999999995, 0.24599999999999997, 0.24399999999999999, 0.23824999999999999, 0.2382500000000001, 0.24549999999999994, 0.21424999999999997, 0.21799999999999994, 0.20200000000000004, 0.19724999999999987, 0.19624999999999995, 0.18700000000000003, 0.18899999999999989, 0.18075000000000002, 0.17675000000000002, 0.17050000000000004, 0.17450000000000002, 0.1717499999999999, 0.16449999999999998, 0.15525000000000003, 0.15024999999999999, 0.14849999999999997, 0.15274999999999997, 0.14024999999999996, 0.14074999999999999, 0.13100000000000001, 0.1235, 0.12924999999999998, 0.11574999999999992, 0.11724999999999999, 0.10725000000000001, 0.098500000000000018, 0.11399999999999995, 0.09425, 0.10250000000000002, 0.092250000000000013, 0.095499999999999988, 0.086749999999999994, 0.089250000000000065, 0.085750000000000007, 0.076249999999999998, 0.070500000000000007, 0.067999999999999991, 0.063999999999999974, 0.062249999999999986, 0.056250000000000008, 0.056250000000000001, 0.056749999999999988, 0.052999999999999999, 0.044499999999999984, 0.04474999999999997, 0.0395, 0.032249999999999973, 0.037749999999999964, 0.031999999999999973, 0.027249999999999983, 0.027249999999999969, 0.023499999999999986, 0.019249999999999993, 0.015749999999999997, 0.015499999999999998, 0.008750000000000006, 0.01125, 0.0080000000000000036, 0.0032500000000000003, 0.0]
    b = [1.0, 0.95650000000000024, 0.92099999999999993, 0.90675000000000039, 0.87725000000000009, 0.81824999999999992, 0.79049999999999965, 0.77375000000000016, 0.74724999999999975, 0.75525000000000009, 0.71325000000000049, 0.70825000000000016, 0.66849999999999998, 0.65724999999999978, 0.6382500000000001, 0.65750000000000031, 0.61624999999999985, 0.6472500000000001, 0.59250000000000003, 0.59449999999999992, 0.56799999999999995, 0.57774999999999999, 0.56950000000000001, 0.56149999999999989, 0.53625000000000012, 0.51849999999999996, 0.49825000000000008, 0.50075000000000014, 0.47950000000000009, 0.48399999999999999, 0.47225000000000017, 0.45674999999999988, 0.44550000000000012, 0.4517500000000001, 0.44950000000000018, 0.43250000000000016, 0.43224999999999997, 0.39850000000000002, 0.40100000000000008, 0.39624999999999999, 0.39775000000000021, 0.38274999999999987, 0.38425000000000009, 0.36475000000000007, 0.36575000000000002, 0.34900000000000003, 0.34449999999999981, 0.32899999999999996, 0.32450000000000012, 0.31300000000000006, 0.31049999999999994, 0.30600000000000011, 0.30399999999999999, 0.27724999999999977, 0.27250000000000008, 0.27175000000000005, 0.26949999999999991, 0.26525000000000004, 0.25974999999999993, 0.25724999999999992, 0.23725000000000004, 0.23775000000000004, 0.22725000000000012, 0.23549999999999996, 0.2205, 0.21399999999999988, 0.21475000000000005, 0.18275000000000002, 0.19450000000000001, 0.185, 0.18375000000000019, 0.16824999999999996, 0.16575000000000004, 0.1515, 0.16674999999999987, 0.15475, 0.13674999999999998, 0.13599999999999998, 0.13424999999999998, 0.13125000000000001, 0.12024999999999995, 0.11225000000000003, 0.099999999999999978, 0.096000000000000002, 0.090750000000000011, 0.089749999999999941, 0.080749999999999975, 0.087499999999999967, 0.07400000000000001, 0.068500000000000019, 0.058749999999999976, 0.058499999999999955, 0.051249999999999983, 0.038999999999999972, 0.036499999999999984, 0.031749999999999966, 0.02424999999999998, 0.017499999999999988, 0.012000000000000002, 0.0075000000000000023, 0.0]
    c = [1.0, 0.97250000000000003, 0.94999999999999996, 0.88749999999999984, 0.87333333333333329, 0.87083333333333313, 0.83999999999999997, 0.80666666666666642, 0.78833333333333333, 0.75833333333333297, 0.73833333333333317, 0.72833333333333328, 0.69583333333333341, 0.66749999999999998, 0.64583333333333326, 0.63333333333333353, 0.64166666666666661, 0.62916666666666676, 0.58583333333333332, 0.57833333333333325, 0.60999999999999988, 0.59083333333333343, 0.57999999999999996, 0.56583333333333319, 0.51249999999999996, 0.50583333333333325, 0.54749999999999999, 0.47833333333333333, 0.51083333333333336, 0.47333333333333322, 0.47166666666666662, 0.45666666666666667, 0.41166666666666657, 0.42416666666666658, 0.45250000000000007, 0.41499999999999998, 0.42083333333333328, 0.38249999999999995, 0.41749999999999998, 0.40250000000000002, 0.36749999999999994, 0.36333333333333345, 0.35249999999999998, 0.36499999999999999, 0.35666666666666669, 0.32166666666666666, 0.33749999999999991, 0.33833333333333343, 0.31833333333333336, 0.32000000000000001, 0.30583333333333323, 0.30250000000000005, 0.25333333333333335, 0.2583333333333333, 0.25083333333333346, 0.255, 0.28083333333333338, 0.26083333333333336, 0.25083333333333335, 0.22333333333333333, 0.21083333333333332, 0.20999999999999996, 0.20499999999999999, 0.19833333333333342, 0.18583333333333341, 0.17666666666666664, 0.19916666666666658, 0.1783333333333334, 0.1883333333333333, 0.16250000000000003, 0.18416666666666667, 0.15083333333333335, 0.16, 0.16333333333333333, 0.13750000000000001, 0.13250000000000001, 0.12333333333333338, 0.11166666666666668, 0.10249999999999999, 0.10999999999999996, 0.10583333333333332, 0.088333333333333361, 0.098333333333333356, 0.090833333333333321, 0.07333333333333332, 0.072499999999999995, 0.07333333333333332, 0.066666666666666666, 0.055833333333333332, 0.058333333333333341, 0.047500000000000001, 0.043333333333333328, 0.054166666666666634, 0.04250000000000001, 0.026666666666666665, 0.022499999999999999, 0.018333333333333333, 0.013333333333333331, 0.011666666666666667, 0.0033333333333333331, 0.00083333333333333328]
    d = [1.0, 0.99544444444444435, 0.99777777777777776, 0.99088888888888893, 0.98455555555555552, 0.97344444444444445, 0.97300000000000009, 0.96466666666666667, 0.95133333333333325, 0.95066666666666677, 0.94300000000000028, 0.93644444444444441, 0.95033333333333336, 0.94099999999999984, 0.8977777777777779, 0.89655555555555577, 0.88788888888888906, 0.8823333333333333, 0.86444444444444457, 0.86122222222222222, 0.85944444444444412, 0.85944444444444446, 0.83844444444444444, 0.8383333333333336, 0.81700000000000017, 0.81499999999999984, 0.81755555555555559, 0.78988888888888864, 0.77144444444444471, 0.76999999999999991, 0.78433333333333333, 0.75588888888888905, 0.76111111111111118, 0.75211111111111117, 0.74766666666666681, 0.74744444444444436, 0.74455555555555575, 0.73455555555555563, 0.72655555555555562, 0.71344444444444421, 0.71311111111111136, 0.69844444444444431, 0.71166666666666667, 0.70377777777777784, 0.70733333333333315, 0.68155555555555547, 0.67733333333333301, 0.67666666666666675, 0.66322222222222227, 0.66055555555555545, 0.65255555555555544, 0.63788888888888906, 0.63122222222222224, 0.63922222222222225, 0.62155555555555564, 0.59988888888888869, 0.59144444444444444, 0.60444444444444467, 0.60099999999999998, 0.58566666666666656, 0.59033333333333338, 0.57055555555555559, 0.55211111111111111, 0.53722222222222205, 0.53644444444444461, 0.53400000000000003, 0.53199999999999992, 0.52433333333333332, 0.48311111111111094, 0.48755555555555569, 0.49866666666666676, 0.46577777777777785, 0.47722222222222216, 0.46599999999999997, 0.42533333333333345, 0.44933333333333331, 0.40422222222222226, 0.41066666666666679, 0.38377777777777777, 0.38588888888888895, 0.38844444444444448, 0.36188888888888876, 0.35133333333333339, 0.33500000000000002, 0.31733333333333325, 0.30277777777777781, 0.28877777777777774, 0.27200000000000002, 0.23966666666666664, 0.23555555555555555, 0.21800000000000005, 0.2098888888888889, 0.1778888888888889, 0.16322222222222224, 0.13900000000000001, 0.12144444444444441, 0.091444444444444398, 0.077000000000000013, 0.045666666666666661, 0.023333333333333327, 0.0]

    e = [1.0, 0.72856321839080451, 0.59290804597701141, 0.55678160919540209, 0.51726436781609186, 0.47637931034482761, 0.4502298850574713, 0.44370114942528738, 0.41986206896551725, 0.4029655172413793, 0.39204597701149424, 0.37698850574712645, 0.36618390804597689, 0.36439080459770118, 0.34603448275862081, 0.33450574712643677, 0.32603448275862074, 0.32320689655172408, 0.30795402298850566, 0.30362068965517236, 0.30194252873563221, 0.29325287356321839, 0.28470114942528729, 0.28109195402298853, 0.27531034482758626, 0.26432183908045975, 0.25879310344827589, 0.25162068965517237, 0.24998850574712647, 0.2400114942528736, 0.23591954022988504, 0.23263218390804599, 0.22832183908045978, 0.22405747126436781, 0.21643678160919536, 0.21001149425287355, 0.20580459770114945, 0.20125287356321833, 0.200367816091954, 0.1940804597701149, 0.19020689655172413, 0.18373563218390807, 0.18062068965517242, 0.17901149425287366, 0.17331034482758617, 0.17091954022988506, 0.16589655172413795, 0.16049425287356323, 0.15728735632183899, 0.1545747126436782, 0.15128735632183907, 0.14503448275862066, 0.14375862068965517, 0.14036781609195403, 0.13337931034482756, 0.13277011494252869, 0.12825287356321843, 0.12514942528735631, 0.12145977011494249, 0.11555172413793102, 0.11522988505747125, 0.11054022988505746, 0.10733333333333332, 0.10427586206896551, 0.10268965517241377, 0.099781609195402257, 0.094149425287356328, 0.090839080459770127, 0.087425287356321865, 0.086172413793103447, 0.082666666666666652, 0.079402298850574718, 0.07736781609195402, 0.072091954022988514, 0.070172413793103461, 0.066655172413793096, 0.063241379310344834, 0.061609195402298846, 0.058126436781609191, 0.054068965517241371, 0.054563218390804601, 0.05013793103448274, 0.048402298850574697, 0.044816091954023002, 0.041689655172413798, 0.039160919540229877, 0.035770114942528734, 0.033885057471264371, 0.030747126436781604, 0.028356321839080451, 0.024839080459770114, 0.022402298850574705, 0.019540229885057471, 0.017735632183908046, 0.015494252873563218, 0.012839080459770115, 0.010379310344827586, 0.0071379310344827597, 0.0052068965517241377, 0.0025632183908045987, 0.0]

    p10 = [1.0, 0.85577777777777797, 0.75277777777777755, 0.68544444444444452, 0.67533333333333334, 0.65099999999999991, 0.58466666666666667, 0.54066666666666674, 0.49888888888888877, 0.47766666666666652, 0.47922222222222222, 0.46611111111111109, 0.42188888888888892, 0.41122222222222221, 0.3813333333333333, 0.37855555555555548, 0.36111111111111105, 0.37311111111111117, 0.32344444444444448, 0.33222222222222214, 0.31766666666666671, 0.32322222222222224, 0.32033333333333347, 0.3166666666666666, 0.30311111111111105, 0.2755555555555555, 0.28166666666666673, 0.28944444444444439, 0.26900000000000007, 0.26177777777777766, 0.25844444444444453, 0.24355555555555547, 0.23288888888888887, 0.2301111111111111, 0.22955555555555557, 0.21888888888888888, 0.22411111111111112, 0.21322222222222226, 0.21155555555555558, 0.20377777777777781, 0.20611111111111108, 0.19433333333333333, 0.19433333333333336, 0.17688888888888901, 0.1831111111111112, 0.17322222222222228, 0.17444444444444454, 0.17255555555555563, 0.16977777777777786, 0.16244444444444453, 0.15500000000000011, 0.15700000000000003, 0.14400000000000002, 0.14055555555555554, 0.14244444444444448, 0.13233333333333336, 0.12811111111111109, 0.12922222222222221, 0.12022222222222217, 0.12022222222222222, 0.11488888888888886, 0.11155555555555552, 0.11099999999999993, 0.10744444444444441, 0.1041111111111111, 0.097777777777777755, 0.092222222222222219, 0.095444444444444429, 0.088111111111111112, 0.088333333333333361, 0.086222222222222214, 0.085555555555555593, 0.077222222222222234, 0.076777777777777792, 0.072555555555555595, 0.071888888888888891, 0.072000000000000022, 0.062999999999999987, 0.059333333333333335, 0.05255555555555555, 0.05599999999999998, 0.053222222222222185, 0.048222222222222222, 0.047111111111111083, 0.041666666666666678, 0.040777777777777781, 0.035666666666666666, 0.033777777777777782, 0.033444444444444443, 0.029555555555555561, 0.027444444444444448, 0.024333333333333318, 0.022333333333333344, 0.020222222222222225, 0.01555555555555556, 0.013111111111111105, 0.01033333333333333, 0.0065555555555555532, 0.0047777777777777784, 0.0013333333333333333, 0.0]


    p1_10 = [1.0, 0.98988888888888893, 0.98277777777777786, 0.96666666666666667, 0.92911111111111111, 0.90133333333333321, 0.86899999999999988, 0.81955555555555537, 0.82133333333333314, 0.80388888888888876, 0.74411111111111128, 0.75088888888888872, 0.69744444444444464, 0.70433333333333348, 0.65144444444444427, 0.6671111111111111, 0.59944444444444467, 0.57711111111111113, 0.54222222222222238, 0.55700000000000005, 0.51944444444444438, 0.46511111111111098, 0.4787777777777778, 0.4821111111111111, 0.44822222222222208, 0.45133333333333325, 0.43455555555555558, 0.42544444444444429, 0.40088888888888896, 0.3915555555555556, 0.3742222222222224, 0.36822222222222228, 0.36355555555555552, 0.34555555555555556, 0.34933333333333327, 0.33088888888888895, 0.32300000000000006, 0.30144444444444446, 0.318, 0.29688888888888881, 0.28422222222222226, 0.28633333333333338, 0.27177777777777778, 0.25955555555555554, 0.26088888888888895, 0.26255555555555554, 0.25488888888888889, 0.24655555555555558, 0.24377777777777773, 0.23000000000000004, 0.21911111111111101, 0.22088888888888886, 0.21366666666666664, 0.19711111111111113, 0.20700000000000002, 0.19411111111111115, 0.19033333333333338, 0.18066666666666678, 0.17933333333333337, 0.18377777777777779, 0.15711111111111117, 0.15733333333333344, 0.16166666666666674, 0.15455555555555564, 0.15300000000000011, 0.15100000000000002, 0.1435555555555556, 0.13588888888888895, 0.12488888888888884, 0.12833333333333335, 0.11399999999999995, 0.11877777777777775, 0.10744444444444441, 0.10244444444444442, 0.10077777777777776, 0.09999999999999995, 0.088111111111111112, 0.09244444444444444, 0.085666666666666683, 0.090222222222222204, 0.077444444444444469, 0.075222222222222246, 0.067666666666666653, 0.068000000000000005, 0.066333333333333341, 0.065333333333333299, 0.054999999999999979, 0.052111111111111115, 0.04222222222222223, 0.043333333333333328, 0.045888888888888868, 0.035555555555555556, 0.028333333333333318, 0.029111111111111102, 0.020000000000000007, 0.017444444444444443, 0.016999999999999998, 0.011888888888888888, 0.0057777777777777758, 0.0024444444444444435, 0.0]

    p2_10 = [1.0, 0.98866666666666669, 0.98844444444444446, 0.94211111111111112, 0.95088888888888889, 0.91400000000000003, 0.88955555555555521, 0.88122222222222224, 0.86877777777777776, 0.88122222222222224, 0.77600000000000013, 0.81544444444444442, 0.74811111111111084, 0.7430000000000001, 0.71711111111111125, 0.73344444444444434, 0.69766666666666632, 0.60966666666666647, 0.57688888888888901, 0.58155555555555527, 0.57677777777777772, 0.53222222222222226, 0.53688888888888886, 0.48022222222222227, 0.46033333333333326, 0.43944444444444436, 0.40666666666666679, 0.43788888888888899, 0.4032222222222222, 0.37011111111111111, 0.34800000000000009, 0.3298888888888889, 0.34377777777777779, 0.33111111111111108, 0.28933333333333328, 0.30499999999999994, 0.31011111111111117, 0.29855555555555557, 0.26666666666666661, 0.25511111111111112, 0.25866666666666666, 0.23788888888888882, 0.23433333333333325, 0.22866666666666668, 0.22177777777777766, 0.22044444444444442, 0.20611111111111113, 0.20644444444444449, 0.1962222222222223, 0.18611111111111114, 0.17344444444444451, 0.17333333333333342, 0.16666666666666674, 0.16688888888888898, 0.15344444444444444, 0.1521111111111112, 0.1476666666666667, 0.14444444444444449, 0.14533333333333334, 0.14011111111111113, 0.13422222222222224, 0.13299999999999998, 0.12466666666666669, 0.11455555555555547, 0.11722222222222217, 0.1113333333333333, 0.10666666666666663, 0.10566666666666663, 0.092999999999999972, 0.089222222222222203, 0.094111111111111118, 0.088555555555555554, 0.08611111111111111, 0.078444444444444483, 0.072888888888888892, 0.06822222222222224, 0.067333333333333356, 0.070333333333333345, 0.060666666666666647, 0.057999999999999961, 0.056222222222222208, 0.055444444444444442, 0.048333333333333318, 0.048999999999999988, 0.047111111111111111, 0.045111111111111109, 0.038666666666666676, 0.033222222222222215, 0.032666666666666663, 0.028666666666666663, 0.028555555555555542, 0.023888888888888894, 0.018555555555555558, 0.019000000000000003, 0.01822222222222223, 0.01233333333333333, 0.0086666666666666628, 0.009111111111111108, 0.006111111111111108, 0.0031111111111111105, 0.0]


    p5 = [1.0, 0.92599999999999993, 0.88800000000000001, 0.85699999999999987, 0.70750000000000013, 0.78650000000000009, 0.67449999999999988, 0.69850000000000012, 0.72049999999999981, 0.54249999999999998, 0.59199999999999997, 0.58250000000000002, 0.57300000000000006, 0.50649999999999995, 0.49099999999999988, 0.43550000000000005, 0.44000000000000006, 0.47699999999999998, 0.41749999999999998, 0.42400000000000004, 0.40400000000000008, 0.39949999999999991, 0.36650000000000005, 0.35649999999999998, 0.37, 0.36499999999999999, 0.34900000000000003, 0.31299999999999994, 0.29449999999999987, 0.3165, 0.3015000000000001, 0.32550000000000007, 0.28499999999999998, 0.30349999999999999, 0.25050000000000006, 0.23499999999999993, 0.29650000000000004, 0.24050000000000002, 0.23199999999999996, 0.24149999999999999, 0.24399999999999999, 0.22000000000000011, 0.22450000000000006, 0.20650000000000002, 0.19400000000000003, 0.19500000000000001, 0.1865, 0.17800000000000002, 0.19500000000000003, 0.20250000000000004, 0.17650000000000005, 0.17650000000000002, 0.17900000000000002, 0.18449999999999997, 0.15400000000000003, 0.14150000000000004, 0.17149999999999996, 0.13599999999999995, 0.1585, 0.12299999999999996, 0.13600000000000001, 0.14599999999999999, 0.11550000000000001, 0.128, 0.1205, 0.1065, 0.12249999999999997, 0.11200000000000002, 0.10250000000000002, 0.097000000000000017, 0.099500000000000047, 0.076000000000000012, 0.096500000000000002, 0.094000000000000028, 0.077499999999999986, 0.057000000000000009, 0.077000000000000013, 0.068500000000000005, 0.074499999999999969, 0.063999999999999987, 0.064999999999999988, 0.049499999999999995, 0.064999999999999974, 0.054999999999999979, 0.037500000000000006, 0.041499999999999995, 0.048499999999999995, 0.035999999999999997, 0.034999999999999989, 0.031999999999999987, 0.0275, 0.025000000000000001, 0.025500000000000009, 0.021000000000000005, 0.015500000000000007, 0.016000000000000004, 0.013000000000000005, 0.0050000000000000001, 0.0049999999999999992, 0.00050000000000000001, 0.0]

    p1_5 = [1.0, 0.99600000000000011, 0.98450000000000004, 0.97599999999999998, 0.96299999999999997, 0.93100000000000005, 0.93649999999999967, 0.93999999999999995, 0.91799999999999993, 0.89499999999999991, 0.86250000000000016, 0.8580000000000001, 0.85449999999999959, 0.85350000000000004, 0.81249999999999989, 0.74799999999999978, 0.74450000000000005, 0.76049999999999995, 0.75399999999999989, 0.70050000000000012, 0.7340000000000001, 0.72049999999999981, 0.67200000000000004, 0.69349999999999989, 0.64899999999999991, 0.64100000000000013, 0.58750000000000002, 0.61250000000000004, 0.61750000000000005, 0.55449999999999999, 0.53199999999999992, 0.53350000000000009, 0.53599999999999992, 0.46550000000000002, 0.49449999999999994, 0.47950000000000015, 0.4265000000000001, 0.42899999999999999, 0.42549999999999988, 0.40350000000000014, 0.39749999999999991, 0.40450000000000008, 0.37450000000000006, 0.33849999999999997, 0.39800000000000002, 0.32900000000000013, 0.33599999999999997, 0.3705, 0.27950000000000003, 0.29199999999999998, 0.26000000000000006, 0.24200000000000002, 0.3115, 0.25349999999999995, 0.27149999999999996, 0.23999999999999999, 0.2495, 0.20450000000000002, 0.21649999999999994, 0.19500000000000006, 0.20449999999999999, 0.19750000000000006, 0.19000000000000003, 0.17600000000000002, 0.17850000000000005, 0.15050000000000002, 0.16899999999999998, 0.13999999999999999, 0.13799999999999998, 0.1535, 0.13900000000000001, 0.11750000000000002, 0.11999999999999998, 0.11100000000000003, 0.10550000000000001, 0.10349999999999999, 0.090499999999999983, 0.080999999999999975, 0.1045, 0.086500000000000007, 0.093999999999999986, 0.062999999999999973, 0.068000000000000005, 0.076499999999999999, 0.063499999999999987, 0.052999999999999992, 0.050999999999999997, 0.050499999999999996, 0.047999999999999959, 0.050000000000000003, 0.044999999999999991, 0.032999999999999995, 0.0275, 0.029499999999999992, 0.025000000000000001, 0.0315, 0.019000000000000003, 0.015000000000000005, 0.0054999999999999997, 0.0040000000000000001, 0.0]

    p2_5 = [1.0, 0.98649999999999993, 0.98750000000000004, 0.96799999999999997, 0.95650000000000002, 0.97399999999999975, 0.93500000000000005, 0.94349999999999978, 0.9474999999999999, 0.92350000000000021, 0.86099999999999999, 0.90049999999999986, 0.87749999999999939, 0.84849999999999981, 0.83550000000000002, 0.79649999999999987, 0.78699999999999992, 0.80899999999999994, 0.76500000000000001, 0.76549999999999996, 0.75050000000000017, 0.77150000000000019, 0.65799999999999992, 0.71050000000000013, 0.67849999999999999, 0.66799999999999982, 0.67100000000000004, 0.60050000000000003, 0.59900000000000009, 0.57500000000000007, 0.49150000000000005, 0.53749999999999998, 0.50199999999999989, 0.5405000000000002, 0.46950000000000003, 0.4375, 0.55099999999999993, 0.39600000000000013, 0.36399999999999999, 0.34400000000000008, 0.40600000000000008, 0.38199999999999995, 0.39949999999999997, 0.35399999999999998, 0.33200000000000007, 0.28350000000000003, 0.26900000000000007, 0.30500000000000005, 0.28750000000000003, 0.25750000000000006, 0.25549999999999995, 0.20599999999999999, 0.20899999999999999, 0.20849999999999999, 0.20899999999999999, 0.20549999999999996, 0.16200000000000001, 0.16300000000000001, 0.16, 0.15600000000000006, 0.14850000000000002, 0.16049999999999998, 0.14050000000000007, 0.12400000000000003, 0.13500000000000001, 0.13700000000000001, 0.12, 0.106, 0.10099999999999998, 0.10850000000000001, 0.098999999999999991, 0.094500000000000028, 0.089999999999999997, 0.083000000000000004, 0.082500000000000004, 0.072000000000000008, 0.07599999999999997, 0.074499999999999969, 0.065499999999999975, 0.060999999999999978, 0.061499999999999985, 0.063499999999999987, 0.041500000000000002, 0.052999999999999999, 0.05050000000000001, 0.045999999999999985, 0.045999999999999999, 0.041499999999999995, 0.029999999999999999, 0.035499999999999997, 0.032500000000000001, 0.026999999999999993, 0.021500000000000005, 0.017500000000000005, 0.013000000000000005, 0.011000000000000003, 0.013000000000000005, 0.0080000000000000019, 0.0060000000000000001, 0.0015000000000000002, 0.0]

    rbh = [0.5, 0.49049999999999999, 0.48099999999999993, 0.46100000000000008, 0.46500000000000002, 0.44600000000000001, 0.43349999999999994, 0.42499999999999993, 0.42199999999999988, 0.41400000000000003, 0.40849999999999986, 0.39349999999999985, 0.39150000000000007, 0.40600000000000003, 0.39449999999999996, 0.38, 0.36899999999999994, 0.36849999999999994, 0.34199999999999997, 0.34450000000000003, 0.33100000000000002, 0.34399999999999992, 0.32450000000000001, 0.33050000000000002, 0.31449999999999995, 0.30649999999999999, 0.3005000000000001, 0.28849999999999998, 0.28600000000000003, 0.28649999999999998, 0.27550000000000002, 0.26349999999999996, 0.27550000000000002, 0.25799999999999995, 0.27000000000000002, 0.23100000000000001, 0.24200000000000008, 0.26149999999999995, 0.24849999999999997, 0.23300000000000004, 0.23999999999999996, 0.23300000000000001, 0.22350000000000006, 0.20449999999999996, 0.20650000000000002, 0.19600000000000006, 0.19950000000000007, 0.21699999999999997, 0.20350000000000004, 0.19149999999999998, 0.18699999999999994, 0.19, 0.18700000000000003, 0.17749999999999999, 0.17399999999999999, 0.16299999999999998, 0.17699999999999996, 0.13449999999999998, 0.1595, 0.14949999999999997, 0.14699999999999996, 0.14649999999999996, 0.14200000000000002, 0.13349999999999998, 0.13000000000000003, 0.10549999999999998, 0.13550000000000001, 0.12049999999999995, 0.13150000000000003, 0.12099999999999998, 0.11949999999999998, 0.11849999999999997, 0.11299999999999998, 0.10949999999999999, 0.10700000000000001, 0.091999999999999971, 0.084500000000000033, 0.085999999999999979, 0.080999999999999975, 0.079500000000000001, 0.076500000000000012, 0.072499999999999995, 0.06649999999999999, 0.078499999999999973, 0.064999999999999988, 0.071000000000000008, 0.06049999999999997, 0.064000000000000015, 0.060999999999999964, 0.0545, 0.057999999999999968, 0.039499999999999987, 0.044999999999999998, 0.041000000000000009, 0.040500000000000001, 0.029500000000000002, 0.034999999999999989, 0.036500000000000012, 0.021500000000000005, 0.021500000000000005, 0.017500000000000005]


    """
    u1 = p5
    u2 = b
    u3 = rbh

    ns = np.arange(0.0, 1.001, 0.01)
    
    rns = [1-x for x in ns]
    plt.plot(ns,u1, 'g-', ns,u2, 'r-', ns, rns, 'm-', ns, u3, 'b-', linewidth=3)
    plt.annotate("Prim", (ns[22],u1[22]), xytext = (65, 5), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("No cascading", (ns[60],rns[60]), xytext = (130, 40), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("Kruskal", (ns[25],u2[25]), xytext = (60, 30), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("RBH", (ns[10],u3[10]), xytext = (30, -50), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))

    plt.xlabel("Score function error rate", size=35)
    plt.ylabel("Prop. correct edges", size=35)
    plt.tick_params(axis='both', labelsize=30)
    plt.subplots_adjust(left=0.19, right=0.94, top=0.94, bottom=0.15)
    plt.show()
    """
    """
    ns = np.arange(0.0, 1.001, 0.01)
    
    rns = [1-x for x in ns]
    plt.plot(ns,e, 'g-', ns,b, 'r-', ns, d, 'b-', ns, rns, 'm-')
    plt.annotate("Prim - 30*30", (ns[20],e[20]), xytext = (40, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Kruskal - 5*5", (ns[60],b[60]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("No cascading", (ns[50],rns[50]), xytext = (40, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim_Correction - 10*10", (ns[40],d[40]), xytext = (100, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Prim 1*25", (ns[60],c[60]), xytext = (-30, -10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Kruskal 1*25", (ns[50],d[50]), xytext = (30, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Cost/score function error rate")
    plt.ylabel("Proportion correct edges")
    plt.show()
    """
    
    
    """
    pbgcc = [1.0, 0.416666666667, 1.0, 0.825, 0.966666666667, 0.571428571429, 0.660714285714, 0.611111111111, 0.494444444444, 0.359090909091, 0.371212121212, 0.304487179487, 0.266483516484, 0.316666666667, 0.322916666667, 0.257352941176, 0.218954248366, 0.191520467836, 0.192105263158]
    ppcc = [1.0, 0.666666666667, 0.916666666667, 1.0, 0.783333333333, 0.738095238095, 0.633928571429, 0.555555555556, 0.672222222222, 0.445454545455, 0.5, 0.38141025641, 0.423076923077, 0.430952380952, 0.63125, 0.4375, 0.34477124183, 0.210526315789, 0.284210526316 ]
    pps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.98989898989899, 0.9916666666666667, 0.993006993006993, 0.9880952380952381, 0.9538461538461539, 0.9910714285714286, 0.9882352941176471, 0.9895833333333334, 0.9845201238390093, 0.9777777777777777, 0.9799498746867168 ]
    ln = len(pps)
    xs = [i*i for i in range(2, ln + 2)]
    plt.plot(xs,ppcc, 'r-H')#,linewidth=4, markersize=12)
    plt.plot(xs, pps, 'b-d')#,linewidth=4, markersize=12)#,xs,pbgcc, 'g-*', ns, c, 'm-', ns, d, 'c-')
    #plt.annotate("Cross-cut - BestGausCost", (xs[12],pbgcc[12]), xytext = (70, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Cross-cut", (xs[10],ppcc[10]), xytext = (60, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Strip-cut", (xs[12], pps[12]), xytext = (30, -25), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")#, size=20)
    plt.ylabel("Proportion correct edges")#, size=20)
    #plt.tick_params(axis='both', labelsize=18)
    a = plt.gca()
    a.set_ylim([0,1.0])
    plt.show()
    """

    
    t = """
2 1.0 1 1.27
3 0.75 2 1.36
4 0.666666666667 3 1.56
5 0.475 7 1.86
6 0.8 3 2.4
7 0.630952380952 10 3.03
8 0.607142857143 20 4.19
9 0.479166666667 30 5.59
10 0.366666666667 48 8.18
11 0.554545454545 40 10.36
12 0.344696969697 75 15.46
13 0.400641025641 83 20.37
14 0.302197802198 109 30.34
15 0.266666666667 134 35.43
16 0.2375 159 56.99
"""

    tp = """
2 1.0 1 1.29
3 0.75 2 1.4
4 0.833333333333 2 1.59
5 0.5 9 2.03
6 0.783333333333 4 3.95
7 0.559523809524 11 3.8
8 0.428571428571 26 6.78
9 0.708333333333 14 8.09
10 0.477777777778 36 15.19
11 0.381818181818 59 20.45
12 0.284090909091 80 34.89
13 0.416666666667 78 60.27
14 0.337912087912 101 424.03
15 0.302380952381 126 129.38
16 0.216666666667 166 213.51
"""

    tpr = """
2 1.0 1 1.28
3 0.666666666667 3 1.38
4 0.833333333333 2 1.59
5 0.55 7 2.16
6 0.6 9 4.55
7 0.607142857143 9 5.22
8 0.598214285714 17 7.45
9 0.729166666667 13 9.2
10 0.483333333333 33 28.88
11 0.513636363636 45 111.71
12 0.462121212121 53 56.5
13 0.410256410256 76 886.62
"""
    """
    pt0 = [y[-1] for y in map(lambda x: x.split(' '), t.strip().split('\n'))]
    pt1 = [y[-1] for y in map(lambda x: x.split(' '), tp.strip().split('\n'))]
    pt2 = [y[-1] for y in map(lambda x: x.split(' '), tpr.strip().split('\n'))]

    xs = [i*i for i in range(2, 17)]
    plt.plot(xs,pt0, 'r-H',xs, pt1, 'b-d', xs[:-3],pt2, 'g-*')#, ns, c, 'm-', ns, d, 'c-')
    #plt.annotate("Cross-cut - BestGausCost", (xs[12],pbgcc[12]), xytext = (70, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim", (xs[10],pt0[10]), xytext = (30, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim - PostProc", (xs[13], pt1[13]), xytext = (60, -25), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim - Runtime & PostProc", (xs[9], pt2[9]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")
    plt.ylabel("Runtime (seconds)")
    a = plt.gca()
    a.set_yscale('log')
    plt.show()
    """

    rbh = """
2 0.25 3 1.32
3 0.333333333333 5 1.46
4 0.541666666667 5 1.61
5 0.35 11 1.86
6 0.3 19 2.38
7 0.333333333333 22 2.91
8 0.366071428571 25 3.73
9 0.333333333333 34 4.92
10 0.283333333333 54 6.23
11 0.231818181818 70 7.95
12 0.231060606061 84 10.23
13 0.189102564103 111 13.2
14 0.266483516484 100 16.33
15 0.2 142 20.48
16 0.197916666667 167 25.82
17 0.152573529412 207 32.04
18 0.161764705882 226 40.24
19 0.12865497076 276 50.57
20 0.123684210526 306 69.82
"""

    prim = """ 
2 1.0 1 1.28
3 0.75 2 1.4
4 0.833333333333 2 1.55
5 0.55 7 1.95
6 0.583333333333 6 2.41
7 0.619047619048 11 3.07
8 0.330357142857 32 4.29
9 0.645833333333 17 5.87
10 0.355555555556 48 7.88
11 0.554545454545 41 10.71
12 0.359848484848 68 15.02
13 0.371794871795 85 21.07
14 0.302197802198 105 30.71
15 0.328571428571 120 42.67
16 0.25625 155 57.27
17 0.248161764706 181 73.12
18 0.307189542484 178 103.71
19 0.299707602339 209 163.02
20 0.194736842105 276 216.48
"""

    recShred = """
2 1.0 1 1.32
3 0.583333333333 3 1.41
4 0.625 4 1.6
5 0.5 9 2.07
6 0.433333333333 13 2.72
7 0.52380952381 12 4.19
8 0.464285714286 23 6.56
9 0.506944444444 23 10.91
10 0.366666666667 42 18.06
11 0.363636363636 55 29.48
12 0.314393939394 74 49.41
13 0.285256410256 96 80.13
14 0.326923076923 94 129.6
15 0.261904761905 132 199.66
16 0.272916666667 141 304.2
17 0.259191176471 160 458.4
18 0.299019607843 159 669.96
19 0.24269005848 213 984.54
20 0.226315789474 242 1509.18
"""

    kruskal = """
2 1.0 1 1.29
3 0.75 2 1.38
4 0.666666666667 3 1.55
5 0.525 8 2.02
6 0.516666666667 8 2.81
7 0.5 10 4.55
8 0.383928571429 21 7.43
9 0.5 20 12.37
10 0.433333333333 35 21.66
11 0.386363636364 51 35.53
12 0.325757575758 69 63.85
13 0.400641025641 74 86.66
14 0.35989010989 84 161.74
15 0.254761904762 130 244.17
16 0.272916666667 139 380.95
17 0.240808823529 165 551.88
18 0.313725490196 150 848.62
19 0.293859649123 184 1310.11
20 0.242105263158 233 1805.65
"""
    
    startPos = 10
    index = 2
    offsets = {2:[(15, 40),(-5, 40),(70, 30),(60, -60)], 1:[(15, 20),(15, 15),(40, -25),(30, 20)], 3:[(25, -65),(70, 30),(40, -25),(25, 20)]}
    indices = {2:[6,4,1,6], 1:[5,3,7,2], 3:[15,16,0,17]}
    rbhS = [y[index] for y in map(lambda x: x.split(' '), rbh.strip().split('\n'))][startPos:]
    primS = [y[index] for y in map(lambda x: x.split(' '), prim.strip().split('\n'))][startPos:]
    recShredS = [y[index] for y in map(lambda x: x.split(' '), recShred.strip().split('\n'))][startPos:]
    kruskalS = [y[index] for y in map(lambda x: x.split(' '), kruskal.strip().split('\n'))][startPos:]

    xs = [i*i for i in range(2, 21)][startPos:]
    plt.plot(xs,kruskalS, 'r-H',xs, rbhS, 'b-d', xs, primS, 'g-*', linewidth=3, markersize=15)#xs, recShredS, 'm-^', linewidth=2, markersize=12)#, ns, d, 'c-')
    print (xs[indices[index][0]],rbhS[indices[index][0]])
    plt.annotate("RBH", (xs[indices[index][0]],rbhS[indices[index][0]]), fontsize=40, xytext = offsets[index][0], textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("Prim", (xs[indices[index][1]],primS[indices[index][1]]), fontsize=40, xytext = offsets[index][1], textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    #plt.annotate("ReconstructShreds", (xs[indices[index][2]], recShredS[indices[index][2]]), fontsize=40, xytext = offsets[index][2], textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("Kruskal", (xs[indices[index][3]], kruskalS[indices[index][3]]), fontsize=40, xytext = offsets[index][3], textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))

    #plt.plot(15*15,800,"mh",markersize=18)
    #plt.annotate("ACO", (15*15, 800), xytext = (100,300),fontsize=40, ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.1), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))

    #plt.plot(15*15,2500,"r*",markersize=22)
    #plt.annotate(r"$HV^2$", (15*15, 2500), xytext = (-35,-39),fontsize=35, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))

    #plt.plot(15*15,4000,"r*",markersize=22)
    #plt.annotate(r"$BV^2$", (15*15, 4000), xytext = (-30,-8),fontsize=35, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))

    #plt.plot(9*9,62,"mh",markersize=18)
    #plt.annotate("ACO", (9*9, 62), xytext = (100,300),fontsize=40, ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.1), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))

    #plt.plot(12*12,280,"mh",markersize=18)
    #plt.annotate("ACO", (12*12, 280), xytext = (100,300),fontsize=40, ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.1), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))


    plt.xlabel("Number of shreds", size=35)
    plt.ylabel("Number of moves left", size=35)
    plt.tick_params(axis='both', labelsize=30)
    #plt.xticks([0,100,200,300,400])
    a = plt.gca()
    #a.set_yscale('log')
    a.set_xlim([140,400])
    a.set_ylim([50,300])
    plt.subplots_adjust(left=0.19, right=0.94, top=0.94, bottom=0.15)
    plt.show()
    
    """
    p = [x[0] for x in [(0.8, 0.0), (0.7749999999999999, 0.0), (0.7598039215686274, 41.270833333333336) , (0.975, 2.45), (0.7711711711711711, 19.470833333333335), (0.6461904761904763, 54.11904761904762), (0.6092032967032969, 92.56473214285714), (0.5316734417344179, 45.14930555555556), (0.6681518151815183, 55.84305555555556), (0.6105067064083457, 55.29090909090909), (0.5205067920585166, 73.20359848484848), (0.4569570135746596, 61.10657051282051), (0.44915490600769725, 57.10302197802198), (0.4560577328276446, 69.84940476190476), (0.5246473735408566, 64.28333333333333), (0.38427991886409757, 77.859375), (0.398109602815484, 60.91053921568628), (0.2955881877806843, 87.99305555555556), (0.3830013781336137, 75.01381578947368)] ]
    q = [x[0] for x in [(0.75, 1.8038133282386566), (0.6375000000000001, 3.3425124410695517), (0.6458333333333333, 3.7734125617064875), (0.825, 3.850164457453277), (0.7438034188034188, 1.3747400868759352), (0.6380772005772007, 3.077451618253361), (0.6554950105042018, 1.8730934144411424), (0.5480492365019806, 2.622829437435161), (0.664290577342048, 2.8995891642396208), (0.5844615384615385, 4.1993985788788875), (0.5716104258178918, 2.581604924076438), (0.47467194438519866, 2.631028637773309), (0.5029462800682958, 2.6249431552190754), (0.49803137443307877, 2.071597279028274), (0.5990971453033893, 1.610507205822009), (0.4543868542575038, 2.234360306630231), (0.447287362649943, 1.7974495531844519), (0.33342581207942107, 2.949557047232156), (0.41095526460619297, 3.9095273918431)] ]
    r = [x[0] for x in [(0.75, 1.5), (0.6375000000000001, 5.583333333333333), (0.6458333333333333, 5.625), (0.6799999999999999, 5.075), (0.710989010989011, 2.1), (0.606677713820571, 2.988095238095238), (0.5789996700434561, 2.1607142857142856), (0.49700418850571304, 2.8819444444444446), (0.5779050567595461, 3.0944444444444446), (0.5208831353831352, 4.213636363636364), (0.5099144917079702, 3.0), (0.43490715884544184, 2.8076923076923075), (0.43365558242670266, 2.697802197802198), (0.4134172319871909,2.461904761904762), (0.5225286406387423, 2.0458333333333334), (0.3619443696207882, 2.4981617647058822), (0.37367085904016933, 2.076797385620915), (0.27045133427066786, 2.953216374269006), (0.32327951086021617, 4.102631578947369)] ]
    
    
    xs = [i*i for i in range(2, 21)]

    plt.plot(xs,p, 'g-*', xs,q, 'r-H', xs, r, 'b-d')#, ns, c, 'm-', ns, d, 'c-')
    plt.annotate("BlackGaussCost", (xs[13],p[13]), xytext = (25, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("ProbScore", (xs[14],q[14]), xytext = (45, 7), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("GaussCost", (xs[7], r[7]), xytext = (22, -20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")
    plt.ylabel(r"Proportion correct edges $(predictedCorrect)$")
    plt.yticks([0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
    plt.show()
    """
    """
    ns = np.arange(0.0, 1.001, 0.01)
    plt.plot(ns,a, 'g-', ns,b, 'r-', ns, rns, 'b-', ns, e, 'm-')#, ns, d, 'c-')
    ma = [a[i] for i in range(len(a)) if i % 5 == 0]
    mb = [b[i] for i in range(len(a)) if i % 5 == 0]
    mrns = [rns[i] for i in range(len(a)) if i % 5 == 0]
    mns = [ns[i] for i in range(len(a)) if i % 5 == 0]
    me = [e[i] for i in range(len(a)) if i % 5 == 0]
    #plt.plot(mns,ma, 'g*', mns,mb, 'rH', mns, mrns, 'bd', mns, me, 'ms')#, ns, d, 'c-')
    plt.annotate("Prim - 5*5", (ns[20],a[20]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Kruskal - 5*5", (ns[60],b[60]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("No cascading", (ns[30],rns[30]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim - 30*30", (ns[30],e[30]), xytext = (30, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Prim 1*25", (ns[60],c[60]), xytext = (-30, -10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    #plt.annotate("Kruskal 1*25", (ns[50],d[50]), xytext = (30, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Cost/score function error rate")
    plt.ylabel("Proportion correct edges")
    plt.show()
    """
    """
    prim = [1.0, 0.75, 0.66666666666666663, 0.47499999999999998, 0.58333333333333337, 0.52380952380952384, 0.625, 0.58333333333333337, 0.42777777777777776]
    prim1 = [1.0, 0.75, 0.66666666666666663, 0.625, 0.66666666666666663, 0.5714285714285714, 0.6785714285714286, 0.75, 0.46666666666666667]
    prim2 = [1.0, 0.75, 0.83333333333333337, 0.59999999999999998, 0.69999999999999996, 0.54761904761904767, 0.6339285714285714, 0.63888888888888884, 0.41666666666666669]

    xs = [x**2 for x in range(2,11)]
    plt.plot(xs, prim, 'b-*', xs, prim1, 'r-H', xs, prim2, 'g-d')
    plt.annotate("Prim", (xs[6],prim[6]), xytext = (10, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim - PostProc", (xs[7],prim1[7]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.annotate("Prim - RunTime & PostProc", (xs[4],prim2[4]), xytext = (60, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds")
    plt.ylabel("Proportion correct edges")

    a = plt.gca()
    a.set_ylim([0.4,1.0])
    plt.show()
    """

  
  elif "8" in arg:
    cutShreds(30,1,Image.open("SampleDocs/p01.png"))
    #im = Image.open("SampleDocs/p01.png")
    #im = Image.open("cmon")
    #im = imgRotate(im, -10)
    #im = im.rotate(10, Image.BICUBIC, expand=True)
    #im = im.rotate(10, Image.BICUBIC, expand=True)
    #im = im.convert("1")
    #im.save("cmon1", "PNG")
  elif "9" in arg:
    """
    im = Image.open("SampleDocs/p01.png").convert("1")
    prior, prxl, pryl, prxr, pryr = getPrediction([im])
    for i in range(2,10):
      scrambleImage(i,1,im)
      imgs = processScan(Image.open("scrambled"))

      page = pages.ImagePage(i, 1, imgs, None, True)
      page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
      (pPos, pEdges) = search.picker("kruskal", page)
      gp = page.calcGroups(pPos)
      groupP = (len(gp),sorted(gp, reverse = True) )
      corrP = page.calcCorrectEdges(pPos)
      print i, corrP
      print groupP
    """
    test="scrambledRot"    
    p01_0 = "SampleDocs/p01Strips0_ptb4.png"
    #p01_1 = "SampleDocs/p01Strips1_ptb4.png"
    #p01 = "SampleDocs/p01_Shreds.png"
    processScan(Image.open(p01_0),(255,0,127,0))
    #extractShred(Image.open("wipBack1"))
    #orient1(Image.open("wip0"))
    #orient2(Image.open("wip0"))
  elif "a" in arg:
    imgs = [[]]
    for s in range(10):
      imgs[0].append(Image.open("wipRot" + str(s)).convert("1"))

    #im = Image.open("SampleDocs/p01.png").convert("1")
    prior, prxl, pryl, prxr, pryr = getPrediction4(imgs[0])

    page = pages.ImagePage(10, 1, imgs, None, True)
    page.setCost("prediction", (prxl,prxr),(pryl,pryr), prior)
    #page.setCost("blackGaus")
    (pPos, pEdges) = search.picker("prim", page)
    #gp = page.calcGroups(pPos)
    groupP = (len(gp),sorted(gp, reverse = True) )
    corrP = page.calcCorrectEdges(pPos)
    print corrP
    print groupP

  elif "b" in arg:
    minInd = 0
    noRow = [0.47658037242297213, 0.49462497488831275, 0.47456038732868699, 0.4577243257949779, 0.38022948376897703, 0.39844405316105758, 0.33730559887765532, 0.35500473869091898, 0.4049667472915936, 0.37398131139288998, 0.27935091212040064][minInd:]

    row = [max(x) for x in zip([0.48184353031770905, 0.5034886112519491, 0.4819714545223629, 0.45598821468386685, 0.38699586553535881, 0.40893963396913829, 0.34238992427448067, 0.34700094266031156, 0.41984430684254576, 0.38252607601433813, 0.28751009684692286], [0.47073241920659792, 0.50803406579740362, 0.4819714545223629, 0.46560359929925144, 0.37875410729360054, 0.40893963396913829, 0.344473257607814, 0.34516270736619392, 0.43455018919548688, 0.38739937036131666, 0.30571185123288758])][minInd:]

    xs = [x**2 for x in range(10,21)][minInd:]
    print len(xs), len(noRow)
    plt.plot(xs, noRow, 'r-^', linewidth=3, markersize=15)
    plt.plot(xs, row, 'b-*', linewidth=3, markersize=20)
    plt.annotate("ProbScore", (xs[6],noRow[6]), xytext = (50, -70), fontsize=40, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("ProbScore+RowScore", (xs[8],row[8]), xytext = (103, 82), fontsize=35, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    #plt.annotate("Prim - RunTime & PostProc", (xs[4],prim2[4]), xytext = (40, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("Number of shreds",size=35)
    plt.ylabel("Prop. correct edges",size=35)
    plt.xticks([100,150,200,250,300,350,400])
    plt.yticks([0.3,0.35,0.4,0.45,0.5])
    plt.tick_params(axis='both', labelsize=30)
    a = plt.gca()
    a.set_ylim([0.27,0.52])
    a.set_xlim([100,400])
    plt.subplots_adjust(left=0.19, right=0.94, top=0.94, bottom=0.15)
    plt.show()
  elif "c" in arg:
    #orient = {25: [0.024, 0.045999999999999999, 0.049333333333333333, 0.042999999999999997, 0.065600000000000006, 0.062666666666666662, 0.073714285714285718, 0.076499999999999999, 0.076888888888888896, 0.069199999999999998, 0.073090909090909095, 0.046666666666666669, 0.046153846153846156, 0.047142857142857146, 0.049333333333333333, 0.050250000000000003, 0.050823529411764705, 0.052444444444444446, 0.065263157894736842, 0.066799999999999998, 0.068380952380952376, 0.070181818181818179, 0.10243478260869565, 0.10333333333333333, 0.10384, 0.10507692307692308, 0.10355555555555555, 0.104, 0.10524137931034483, 0.10546666666666667], 10: [0.0, 0.0, 0.01, 0.012500000000000001, 0.028000000000000001, 0.026666666666666668, 0.035714285714285712, 0.025000000000000001, 0.032222222222222222, 0.032000000000000001, 0.035454545454545454, 0.02, 0.014615384615384615, 0.014999999999999999, 0.017999999999999999, 0.015625, 0.014705882352941176, 0.016111111111111111, 0.015789473684210527, 0.016, 0.015238095238095238, 0.015909090909090907, 0.017826086956521738, 0.017500000000000002, 0.016799999999999999, 0.016923076923076923, 0.016666666666666666, 0.016428571428571428, 0.016551724137931035, 0.017999999999999999], 20: [0.035000000000000003, 0.042500000000000003, 0.028333333333333332, 0.026249999999999999, 0.050000000000000003, 0.051666666666666666, 0.054285714285714284, 0.073124999999999996, 0.061111111111111109, 0.058999999999999997, 0.05909090909090909, 0.035416666666666666, 0.031153846153846153, 0.030357142857142857, 0.033000000000000002, 0.033125000000000002, 0.033235294117647057, 0.033888888888888892, 0.045263157894736845, 0.045499999999999999, 0.045476190476190476, 0.046818181818181821, 0.06347826086956522, 0.064375000000000002, 0.064600000000000005, 0.065576923076923074, 0.065185185185185179, 0.065892857142857142, 0.066724137931034488, 0.068000000000000005], 5: [0.0, 0.0, 0.0, 0.0050000000000000001, 0.012, 0.013333333333333334, 0.0057142857142857143, 0.01, 0.0066666666666666671, 0.016, 0.0072727272727272727, 0.0066666666666666671, 0.0061538461538461538, 0.0057142857142857143, 0.0093333333333333341, 0.01, 0.0094117647058823521, 0.0088888888888888889, 0.010526315789473684, 0.01, 0.0095238095238095247, 0.0090909090909090905, 0.0086956521739130436, 0.0083333333333333332, 0.0080000000000000002, 0.0076923076923076927, 0.0074074074074074077, 0.0071428571428571426, 0.0075862068965517242, 0.0093333333333333341], 15: [0.02, 0.016666666666666666, 0.037777777777777778, 0.041666666666666664, 0.048000000000000001, 0.058888888888888886, 0.075238095238095243, 0.069166666666666668, 0.06222222222222222, 0.070000000000000007, 0.055757575757575756, 0.037777777777777778, 0.033333333333333333, 0.035714285714285712, 0.038666666666666669, 0.033333333333333333, 0.033333333333333333, 0.033703703703703701, 0.038596491228070177, 0.039, 0.038412698412698412, 0.039393939393939391, 0.054202898550724639, 0.055555555555555552, 0.054933333333333334, 0.055128205128205127, 0.054320987654320987, 0.053095238095238098, 0.053103448275862067, 0.055111111111111111], 70: [0.16714285714285715, 0.15642857142857142, 0.15952380952380951, 0.15392857142857144, 0.15485714285714286, 0.14523809523809525, 0.14163265306122449, 0.14607142857142857, 0.14428571428571429, 0.13642857142857143, 0.13844155844155845, 0.13, 0.13076923076923078, 0.13255102040816327, 0.13104761904761905, 0.13312499999999999, 0.13394957983193279, 0.13293650793650794, 0.16541353383458646, 0.16585714285714287, 0.16646258503401359, 0.16688311688311688, 0.21254658385093167, 0.21261904761904762, 0.21154285714285714, 0.21164835164835163, 0.21111111111111111, 0.21158163265306124, 0.21157635467980296, 0.20871428571428571], 40: [0.1825, 0.14499999999999999, 0.1275, 0.10875, 0.111, 0.10541666666666667, 0.10000000000000001, 0.1065625, 0.097500000000000003, 0.089749999999999996, 0.093181818181818185, 0.073749999999999996, 0.074230769230769225, 0.073392857142857149, 0.074166666666666672, 0.071406250000000004, 0.070294117647058826, 0.071388888888888891, 0.096710526315789469, 0.097125000000000003, 0.096666666666666665, 0.097840909090909089, 0.14586956521739131, 0.14531250000000001, 0.14480000000000001, 0.14538461538461539, 0.14481481481481481, 0.14455357142857142, 0.14491379310344826, 0.14391666666666666], 80: [0.20624999999999999, 0.185, 0.18041666666666667, 0.16125, 0.152, 0.14624999999999999, 0.13660714285714284, 0.13500000000000001, 0.13416666666666666, 0.12862499999999999, 0.12806818181818183, 0.11458333333333333, 0.11336538461538462, 0.11321428571428571, 0.11241666666666666, 0.11874999999999999, 0.11941176470588236, 0.11791666666666667, 0.14348684210526316, 0.14299999999999999, 0.14291666666666666, 0.14335227272727272, 0.18461956521739131, 0.18359375, 0.18354999999999999, 0.18326923076923077, 0.18240740740740741, 0.18272321428571428, 0.18301724137931036, 0.17970833333333333], 50: [0.14000000000000001, 0.122, 0.10933333333333334, 0.098500000000000004, 0.106, 0.095000000000000001, 0.098857142857142852, 0.099000000000000005, 0.10044444444444445, 0.092999999999999999, 0.090363636363636368, 0.074499999999999997, 0.073692307692307696, 0.074714285714285719, 0.075866666666666666, 0.081625000000000003, 0.081529411764705878, 0.083111111111111108, 0.11810526315789474, 0.11890000000000001, 0.11866666666666667, 0.12045454545454545, 0.18226086956521739, 0.18174999999999999, 0.18104000000000001, 0.18092307692307694, 0.17962962962962964, 0.17935714285714285, 0.17965517241379311, 0.17773333333333333], 90: [0.24888888888888888, 0.22555555555555556, 0.2088888888888889, 0.18861111111111112, 0.17577777777777778, 0.1711111111111111, 0.15777777777777777, 0.15916666666666668, 0.15135802469135803, 0.14433333333333334, 0.14484848484848484, 0.13305555555555557, 0.13059829059829059, 0.13007936507936507, 0.12785185185185186, 0.13104166666666667, 0.13098039215686275, 0.12919753086419752, 0.15222222222222223, 0.15183333333333332, 0.15153439153439152, 0.15151515151515152, 0.18386473429951691, 0.18291666666666667, 0.18244444444444444, 0.18269230769230768, 0.18164609053497943, 0.1823015873015873, 0.18195402298850574, 0.17892592592592593], 60: [0.16166666666666665, 0.12583333333333332, 0.12055555555555555, 0.10875, 0.10766666666666666, 0.11333333333333333, 0.10404761904761904, 0.10270833333333333, 0.10277777777777777, 0.098166666666666666, 0.10090909090909091, 0.090416666666666673, 0.090256410256410263, 0.088928571428571426, 0.088666666666666671, 0.096458333333333326, 0.095784313725490192, 0.096018518518518517, 0.12780701754385965, 0.12691666666666668, 0.12619047619047619, 0.12696969696969698, 0.17376811594202898, 0.17347222222222222, 0.17380000000000001, 0.1742948717948718, 0.17314814814814813, 0.17410714285714285, 0.17448275862068965, 0.17194444444444446], 30: [0.11666666666666667, 0.098333333333333328, 0.093333333333333338, 0.088333333333333333, 0.104, 0.10166666666666667, 0.10380952380952381, 0.10208333333333333, 0.10481481481481482, 0.096666666666666665, 0.085757575757575755, 0.066944444444444445, 0.059743589743589745, 0.059999999999999998, 0.061777777777777779, 0.067083333333333328, 0.067843137254901958, 0.067962962962962961, 0.082631578947368417, 0.082000000000000003, 0.082063492063492061, 0.083181818181818176, 0.12028985507246377, 0.12069444444444444, 0.12026666666666666, 0.12051282051282051, 0.12012345679012346, 0.12059523809523809, 0.12103448275862069, 0.12077777777777778]}
    orient = {1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10: [0.02, 0.023333333333333334, 0.034000000000000002, 0.052857142857142859, 0.06222222222222222, 0.055454545454545458, 0.043846153846153847, 0.043999999999999997, 0.032352941176470591, 0.033157894736842108], 100: [0.23200000000000001, 0.20066666666666666, 0.155, 0.12814285714285714, 0.11144444444444444, 0.093636363636363643, 0.080384615384615388, 0.070599999999999996, 0.05464705882352941, 0.054473684210526313], 50: [0.154, 0.13200000000000001, 0.12759999999999999, 0.11514285714285714, 0.10511111111111111, 0.089272727272727267, 0.07923076923076923, 0.066799999999999998, 0.051294117647058823, 0.051473684210526317]}

    unknowns = {1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10: [0.0, 0.0, 0.002, 0.012857142857142857, 0.012222222222222223, 0.0081818181818181825, 0.015384615384615385, 0.014666666666666666, 0.013529411764705882, 0.015263157894736841], 100: [0.025999999999999999, 0.026333333333333334, 0.032399999999999998, 0.037999999999999999, 0.046555555555555558, 0.055909090909090908, 0.055538461538461537, 0.062333333333333331, 0.067176470588235296, 0.065631578947368416], 50: [0.02, 0.021999999999999999, 0.026800000000000001, 0.032571428571428571, 0.035555555555555556, 0.040909090909090909, 0.036307692307692305, 0.040533333333333331, 0.045294117647058825, 0.045894736842105266]}

    for k in orient:
      orient[k] = map(lambda (x,y): 1.0 -(x+y), zip(orient[k],unknowns[k]))

    o1 = orient[1]
    o2 = orient[10]
    o3 = orient[50]
    o4 = orient[100]
    xs = range(10,201,20)

    plt.plot(xs, o1, 'b-*', xs, o2, 'g-d', xs, o3, 'r-H', xs, o4, 'm-^', linewidth=3, markersize=15)
    plt.annotate("0 Horizontal Cuts", (xs[4],o1[4]), xytext = (80, 20), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("10 Horizontal Cuts", (xs[6],o2[6]), xytext = (80, 20), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("50 Horizontal Cuts", (xs[3],o3[3]), xytext = (88, 32), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.annotate("100 Horizontal Cuts", (xs[7],o4[7]), xytext = (100, -60), fontsize=25, textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3), arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', lw=2))
    plt.xlabel("Number of vertical cuts", size=30)
    plt.ylabel("Prop. correct orientations", size=30)
    plt.tick_params(axis='both', labelsize=30)
    a = plt.gca()
    a.set_ylim([0.7,1.05])
    plt.subplots_adjust(left=0.19, right=0.94, top=0.94, bottom=0.15)
    plt.show()
  else:
    print 'Unrecognized argument'
