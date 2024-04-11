import random as rnd

# for this example: x must be between -10 and 10 included them

#class Particle:
#    def __init__(self,konum, velocity, pBest)
gBestX = 0
gBestVal = 0
c1 = 1
c2 = 1
iteration = 500

suruX = [-9.6, -6, -2.6, -1.1, 0.6, 2.3, 2.8, 8.3, 10]
velocity = [0,0,0,0,0,0,0,0,0]
pBest = [-9.6,-6,-2.6,-1.1,0.6,2.3,2.8,8.3,10]

def mainFunction(x):
    return -(x**2) +5 * x + 20

def calculateNextVelocity(x, v, pBest, r1, r2):
    return v + c1*r1*(pBest - x) + c2 * r2 * (gBestX - x)

def calculateNextX(x,v):
    return x + v

def calculateGBest():
    best = 0
    bestIndex = 0
    for i in range(len(suruX)):
        val = mainFunction(suruX[i])
        if(val > best):
            best = val
            bestIndex = i
    return suruX[bestIndex], best

def printParticles():
    print("Konum\tFitness\tVelocity\tPbest")
    for i in range(len(suruX)):
        print("%.2f" % suruX[i], end= "\t")
        print("%.2f" % mainFunction(suruX[i]), end= "\t")
        print("%.2f" % velocity[i], end= "\t")
        print("%.2f" % pBest[i], end= "\n")


tempGBestX, tempBest = calculateGBest()
if(tempBest > gBestVal):
    gBestVal = tempBest
    gBestX = tempGBestX

epok = 0
while(epok < iteration):
    r1 = rnd.random()
    r2 = rnd.random()

    for i in range(len(suruX)):
        v1 = calculateNextVelocity(suruX[i], velocity[i], pBest[i], r1, r2)
        x1 = calculateNextX(suruX[i], v1)

        if(x1 >= -10 and x1 <= 10):
            suruX[i], velocity[i] = x1, v1

            parcacikDeger = mainFunction(suruX[i])
            if (parcacikDeger > pBest[i]):
                pBest[i] = parcacikDeger

    tempGBestX, tempBest = calculateGBest()
    if (tempBest > gBestVal):
        gBestVal = tempBest
        gBestX = tempGBestX

    printParticles()

    epok = epok +1
