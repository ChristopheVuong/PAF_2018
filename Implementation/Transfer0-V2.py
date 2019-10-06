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

testCas1 = cas1(sets.sourceSet,sets.destinationSet,[5.3,3.0,1.9916,1.0],
    [
        [1.80318467e-03,3.69500801e-04,1.74728766e-04,2.17909585e-03,
            2.34596419e-03,2.06409958e-03,1.08679805e-03,2.15603061e-04,
            1.07081171e-03,9.72092232e-04,1.52075016e-04,2.44237650e-03,
            7.21737319e-04,1.63475578e-03,3.73345554e-04,4.67092163e-04,
            2.52883226e-03,9.06872785e-04,2.46745542e-03,1.32599625e-03,
            2.09406792e-03,1.14006480e-03,1.06408316e-03,4.39259652e-04,
            6.38176700e-04,1.61661203e-03,1.34572726e-03,2.76943595e-04,
            8.99010326e-04,2.62688262e-04],
        [2.95487642e-03,1.22713671e-03,2.74203087e-03,7.49569315e-04,
            1.09950908e-03,2.90808528e-03,1.73381978e-03,1.23993322e-03,
            4.76891155e-04,7.25081939e-04,1.84055995e-03,2.68830474e-03,
            2.75226261e-03,1.96515565e-03,8.16422340e-04,4.66472606e-04,
            4.27856013e-04,2.54965752e-03,1.20567135e-03,1.22235401e-03,
            1.56356801e-03,2.14585783e-03,2.39180656e-04,2.05797362e-03,
            8.34088404e-04,6.40645645e-04,3.94859958e-04,2.92276338e-03,
            8.60203226e-04,9.36140007e-04],
        [5.32470267e-04,2.57697518e-03,3.47396770e-04,5.61117087e-04,
            2.25233869e-03,2.86112293e-03,2.86039586e-03,2.61227421e-03,
            8.58972145e-05,1.25644243e-03,4.86494224e-04,2.76666311e-03,
            2.91915839e-03,1.15935153e-03,2.57397754e-03,2.51759353e-03,
            1.64115939e-03,2.89771097e-03,2.99687227e-03,2.89262665e-03,
            1.49423569e-04,1.86899183e-03,4.48120125e-04,2.46336093e-03,
            2.01503741e-03,6.10762874e-04,3.04255340e-04,1.08454793e-04,
            8.91347977e-05,2.44882777e-03],
        [4.08291910e-04,4.94694587e-04,4.11578448e-04,5.34523476e-05,
            5.26523926e-04,1.62898948e-03,2.40447861e-05,1.34493555e-04,
            8.36522082e-04,1.06759561e-03,8.96632129e-04,2.40765159e-03,
            1.72080071e-05,2.02185899e-03,1.64396844e-03,5.66509605e-04,
            3.24434229e-04,5.23939731e-05,1.87545676e-03,6.95257487e-04,
            1.22784295e-04,2.55160079e-03,2.59709752e-03,7.71727432e-05,
            1.28915098e-03,1.52767904e-03,2.28114495e-03,2.13746466e-04,
            2.91384244e-03,2.13074924e-03]
    ]
)
testCas1.transfer()

min,projMatrix,projVector = testCas1.getMinProj()

print(min)
print(projMatrix)
print(projVector)

#print(testCas1.test(0,(12.36,21.8,79.78,466.1,0.08772,0.09445,0.06015,0.03745,0.193,0.06404,0.2978,1.502,2.203,20.95,0.007112,0.02493,0.02703,0.01293,0.01958,0.004463,13.83,30.5,91.46,574.7,0.1304,0.2463,0.2434,0.1205,0.2972,0.09261)))

print(testCas1.test(0,(12.25,17.94,78.27,460.3,0.08654,0.06679,0.03885,0.02331,0.197,0.06228,0.22,0.9823,1.484,16.51,0.005518,0.01562,0.01994,0.007924,0.01799,0.002484,13.59,25.22,86.6,564.2,0.1217,0.1788,0.1943,0.08211,0.3113,0.08132)))