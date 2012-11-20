import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

if __name__ == "__main__":
  with open("nnPickled",'r') as f:
    (netX,netY) = pickle.load(f)

  rs = []
  print "BlackGaus"
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              data1 = (i1,i2,i3)
              data2 = (i4,i5,i6)
              rez = 0
              x = 1
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
              rs.append( (data1, data2, rez))

  print sorted(rs, key=lambda x: x[2])

  rs = []
  print "X"
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              rs.append((n, netX.activate(n)))

  print sorted(rs, key=lambda x: x[1])
  rs = []
  print "Y"
  for i1 in [0,1]:
    for i2 in [0,1]:
      for i3 in [0,1]:
        for i4 in [0,1]:
          for i5 in [0,1]:
            for i6 in [0,1]:
              n = (i1,i2,i3,i4,i5,i6)
              rs.append((n, netY.activate(n)))
  print sorted(rs, key=lambda x: x[1])
