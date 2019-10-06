import numpy as np
import adaboost
import sets

class cas1:
    def __init__(self,sourceSet,destinationSet,vect=[],matrix=[]):
        self.sourceSet = sourceSet
        self.destinationSet = destinationSet
        self.sourceEstimators = adaboost.adaBoost(sourceSet).generateDecisionStump(10)
        self.destinationEstimators = []
        self.projMatrix = []
        self.projVector = []
        self.matrix = []
        self.vect = []

    def getDestinationEstimators(self):
        return self.destinationEstimators
        
    def getMinProj(self):
        if len(self.projMatrix) != 0:
            min,rank = self.destinationEstimators[0].getError(),0
            for k in range(1,len(self.destinationEstimators)):
                if(self.destinationEstimators[k].getError() < min):
                    min,rank = self.destinationEstimators[k].getError(),k
            return min,self.projMatrix[rank],self.projVector[rank]
        return -1

    def __generateRandomVector(self,n):
        return 10*np.random.rand(n)
        
    def __generatePseudoRandomVector(self,n,vect):
        return vect + (np.random.rand(n) - 0.5*np.ones(n))

    def __generateRandomMatrix(self,n,p):
        return 0.003*np.random.rand(n,p)

    def __generateProjection(self,n,p):
        matrix = self.__generateRandomMatrix(n,p)
        vect = self.__generateRandomVector(n)
        return lambda x,matrix=matrix,vect=vect: vect + matrix.dot(x),matrix,vect
        
    def __generatePsuedoRandomProjection(self,n,p,vect):
        matrix = self.__generateRandomMatrix(n,p)
        vect = self.__
        return lambda x,matrix=matrix,vect=vect: vect + matrix.dot(x),matrix,vect

    def __generateProjectionNoRandomVect(self,n,p,vect):
        matrix = self.__generateRandomMatrix(n,p)
        return lambda x,matrix=matrix,vect=vect: vect + matrix.dot(x),matrix,vect
        
    def __generatePseudoRandomProjection(self,n,p,vect):
        matrix = self.__generateRandomMatrix(n,p)
        vect = self.__generatePseudoRandomVector(n,vect)
        return lambda x,matrix=matrix,vect=vect: vect + matrix.dot(x),matrix,vect

    def __generateProjectionNoRandom(self,vect,matrix):
        return lambda x: vect + matrix.dot(x)

    def transfer(self,numberOfTests=1):
        self.destinationEstimators,self.projMatrix,self.projVector = [],[],[]
        if self.matrix != [] and self.vect != []:
            proj,projMatrix,projVector = self.__generateProjectionNoRandom(np.array(self.vect),np.array(self.matrix))
            Hproj = []
            for i in range(self.sourceEstimators.size):
                Hproj.append(lambda x,S=self.sourceEstimators[i],P=proj: S(P(x)))
            temp = adaboost.adaBoost(self.destinationSet)
            temp.run(np.array(Hproj))
            self.destinationEstimators.append(temp)
            self.projMatrix.append(projMatrix)
            self.projVector.append(projVector)
        elif self.vect != []:
            for k in range(numberOfTests):
                proj,projMatrix,projVector = self.__generateProjectionPseudoRandomVect(len(self.sourceSet[0][0]),len(self.destinationSet[0][0]),np.array(self.vect))
                Hproj = []
                for i in range(self.sourceEstimators.size):
                    Hproj.append(lambda x,S=self.sourceEstimators[i],P=proj: S(P(x)))
                temp = adaboost.adaBoost(self.destinationSet)
                temp.run(np.array(Hproj))
                self.destinationEstimators.append(temp)
                self.projMatrix.append(projMatrix)
                self.projVector.append(projVector)
        else:
            for k in range(numberOfTests):
                proj,projMatrix,projVector = self.__generateProjection(len(self.sourceSet[0][0]),len(self.destinationSet[0][0]))
                Hproj = []
                for i in range(self.sourceEstimators.size):
                    Hproj.append(lambda x,S=self.sourceEstimators[i],P=proj: S(P(x)))
                temp = adaboost.adaBoost(self.destinationSet)
                temp.run(np.array(Hproj))
                self.destinationEstimators.append(temp)
                self.projMatrix.append(projMatrix)
                self.projVector.append(projVector)

    def test(self,i,x):
        return self.destinationEstimators[i].test(x)

testCas1 = cas1(sets.sourceSet,sets.destinationSet,[5.3,3.0,1.9916,1.0])
testCas1.transfer(100)

min,projMatrix,projVector = testCas1.getMinProj()

print(min)
print(projMatrix)
print(projVector)

#print(testCas1.test(0,(12.36,21.8,79.78,466.1,0.08772,0.09445,0.06015,0.03745,0.193,0.06404,0.2978,1.502,2.203,20.95,0.007112,0.02493,0.02703,0.01293,0.01958,0.004463,13.83,30.5,91.46,574.7,0.1304,0.2463,0.2434,0.1205,0.2972,0.09261)))

#print(testCas1.test(0,(15.37,22.76,100.2,728.2,0.092,0.1036,0.1122,0.07483,0.1717,0.06097,0.3129,0.8413,2.075,29.44,0.009882,0.02444,0.04531,0.01763,0.02471,0.002142,16.43,25.84,107.5,830.9,0.1257,0.1997,0.2846,0.1476,0.2556,0.06828)))