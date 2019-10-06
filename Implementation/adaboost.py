import numpy as np
import sys
import math
import sets

def getError(set,estimator):
    error = 0
    for k in range(set.shape[0]):
        error += abs(estimator(set[k][:set.shape[1]-1]) - set[k][set.shape[1]-1])
    return error * 0.5 / set.shape[0]
    
def getMinMax(set):
    min = [float('inf')]*(set.shape[1]-1)
    max = [-float('inf')]*(set.shape[1]-1)
    for k in range(set.shape[1]-1):
        for i in range(set.shape[0]):
            if set[i][k] > max[k]:
                max[k] = set[i][k]
            if set[i][k] < min[k]:
                min[k] = set[i][k]
    return min,max

class adaBoost:
    def __init__(self,set):
        self.set = set
        self.H = []
        self.funSet = []
        self.alphaSet = []
        self.bordersSet = False
        self.min = False
        self.max = False
        self.estimator = lambda x: 0
        
    def getEstimator(self):
        return self.estimator

    def generateDecisionStump(self,n):
        H = []
        self.__findBorders()
        for k in range(self.set.shape[1]-1):
            fact = (self.max[k] - self.min[k])/(n+1)
            for i in range(n):
                H.append(lambda x,k=k,i=i: 2*(x[k] < self.min[k] + fact*(i+0.5))-1)
        return np.array(H)
        
    def __findBorders(self):
        if not self.bordersSet:
            self.bordersSet = True
            self.min,self.max = getMinMax(self.set)
        
    def runDecisionStump(self,n):
        self.run(self.generateDecisionStump(n))

    def run(self,H):
        w = np.ones(self.set.shape[0]) / self.set.shape[0]
        for k in range(H.size):
             H[k] = lambda x,h=H[k],sign=2*self.__isValid(H[k])-1: sign * h(x)
        self.H,self.funSet,self.alphaSet = H,[],[]
        for k in range(H.size):
            errorsArray = np.array([a[self.set.shape[1]-1] != H[0](a[:self.set.shape[1]-1]) for a in self.set])
            min,minErrorsArray,rank = (errorsArray.dot(w.T)).sum(),errorsArray,0
            
            for i in range(1,H.size-k):
                errorsArray = np.array([a[self.set.shape[1]-1] != H[i](a[:self.set.shape[1]-1]) for a in self.set])
                sumError = (errorsArray*w).sum()
                if(sumError < min):
                    min,minErrorsArray,rank = sumError,errorsArray,i

            if min != 0 and min<1:
                alpha = 0.5*math.log((1-min)/min)
                w = np.array([w[i]*math.exp(alpha) if value == 1 else w[i]*math.exp(-alpha) for i,value in enumerate(minErrorsArray)])
                w /= w.sum()
                self.funSet.append(H[rank])
                self.alphaSet.append(alpha)
                H[rank],H[H.size-k-1] = H[H.size-k-1],H[rank]
            elif min == 0:
                w = np.zeros(minErrorsArray.size)
                self.funSet.append(H[rank])
                self.alphaSet.append(sys.float_info.max)
                break
            else:
                break
                
        self.estimator = lambda x: np.sign(sum(self.alphaSet[i]*h(x) for i,h in enumerate(self.funSet)))
            
    def __isValid(self,h):
        return np.array([abs(h(self.set[k][:self.set.shape[1]-1]) - self.set[k][self.set.shape[1]-1]) * 0.5 for k in range(self.set.shape[0])]).sum() / self.set.shape[0] < 0.5
                
    def getError(self):
        return getError(self.set,self.estimator)
        
    def estimate(self,x):
        return self.estimator(x)

'''test = adaBoost(sets.destinationSet)
test.runDecisionStump(1)
result = test.estimate((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.00000e+01/-2.13246e+00))'''
#result = test.estimate((12.9,20.65,87.14,632.8,0.1152,0.1296,0.0771,0.05003,0.1995,0.08539,0.2962,0.6938,3.021,27.03,0.09017,0.07741,0.02989,0.0561,0.06927,0.010023,16.15,18.51,96.28,789.6,0.5924,0.4917,0.1142,0.06242,0.5727,0.4036))

'''test = adaBoost(sets.sourceSet)
test.runDecisionStump(10)
result = test.estimate((5.3,3.0,1.9916,1.0))
print(result)'''