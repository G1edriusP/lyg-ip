import json
import numpy as np
import time
from multiprocessing import Pool
import multiprocessing



def readFile(fileName):
  with open(fileName) as file:
    data = json.load(file)

    for item in range(len(data["points"])):
      data["points"][item] = np.array([item, data["points"][item]["from"], data["points"][item]["to"]])
      
    return data['points']


def gradientDescent(x):
  eps = 1e-10

  startingStep = 0.1
  step = startingStep

  iterCount = 1000

  index = x[0]
  startX = np.array([x[1], x[2]])
  x = np.array([x[1], x[2]])
  
  for i in range(1, iterCount + 1):
      grad = getGradient(x)
      precisionF = getPrecision(x)

      for j in range(1, 30):
        deltaX = grad / np.linalg.norm(grad) * step

        arr = [x[q] - float(np.transpose(deltaX)[q][0]) for q in range(len(x))]
        x = np.array(arr)

        newPrecisionF = getPrecision(x)
        if newPrecisionF > precisionF:
          arr = [x[q] + float(np.transpose(deltaX)[q][0]) for q in range(len(x))]
          x = np.array(arr)
          step /= 10
        else:
          precisionF = newPrecisionF 

      step = startingStep

      precision = np.linalg.norm(precisionF)
      # print('iteracija %d  tikslumas %g' %(i,tikslumas));
      if precision < eps:
        print('Nr. {0:2.0f}, Sprendinio taško x = {1:.5f} ir y = {2:.5f}'.format(index + 1, x[0], x[1]))
        break
      elif i == iterCount:
        print('Nr. {0:2.0f} - Tikslumas nepasiektas intervale '.format(index + 1), startX)
        break


def getGradient(x):
  return np.dot(np.transpose(f(x)), df(x))


def getPrecision(x):
  return (np.dot(np.transpose(f(x)), f(x))) / 2


def f(x):
  a = np.array(
    [[ (x[0]**2+x[1]**2)/5 - 4*np.sin(2* x[0]) - 4 ],
      [ 100/(x[0]**2 + x[1]**2+5)-x[0]-x[1] ]])
  
  a.shape = (2, 1)
  
  a = np.matrix(a).astype(np.float)
  
  return a


def df(x):

  s = np.array([
                [ 2*x[0] / 5 - 8 * np.cos(2 * x[0]), 2*x[1] / 5 ],
                [ (-200*x[0] / (x[0]**2 + x[1]**2 + 5)**2) - 1, (-200*x[1] / (x[1]**2 + x[0]**2 + 5)**2) - 1 ]
                ])

  s.shape = (2, 2)
  
  return np.matrix(s).astype(np.float)


def main():
  # Duomenų failo varianto numeris (galimi skaičiai: 1-10)
  variant = 1

  logicalProcesses = [16, 8, 4, 2, 1]

  # Programa veiks tiek kartų, kiek kompiuterio procesorius turi loginių procesorių
  for count in logicalProcesses:
    # Duomenu gavimas
    x = readFile("./{:d}_duomenys.json".format(variant))

    # 
    print("\nNaudotų loginių procesorių kiekis - ", count)
    # Uzdavinio pradzios laikas
    timeStart = time.time()
    
    # Panaudotas procesu baseinas
    with Pool(count) as p:
      p.map(gradientDescent, x)
    
    # Uzdavinio pabaigos laikas
    timeEnd = time.time()

    # 
    print("Praėjo {:.3f} sek.".format(timeEnd - timeStart))
  


if __name__ == "__main__":
    main()