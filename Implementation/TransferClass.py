import time
import numpy as np
from adaboost import *
import sets
#import quadprog #Goldfarb & Idnani 1983 alg.


#import cvxopt #useable for semi-definite G (as it is the case here), should use N.L.Boland 1996 extension of Goldfarb & Idnani 1983 alg. instead

def reduceSample(set,numberOfElements):
    if numberOfElements % 2 == 1:
        return numberOfElements + 1
    numberOfElements /= 2
    reducedSample = []
    numberOfClass1 = 0
    numberOfClass2 = 0
    for k in range(set.shape[0]):
        if numberOfClass1 == numberOfElements and numberOfClass2 == numberOfElements:
            break
        else:
            if set[k][set.shape[1]-1] == -1 and numberOfClass1<numberOfElements:
                reducedSample.append(set[k])
                numberOfClass1 += 1
            elif numberOfClass2<numberOfElements:
                reducedSample.append(set[k])
                numberOfClass2 += 1
    return np.array(reducedSample)
    
def shortenDimension(set,numberOfDimension):
    return np.delete(set,np.s_[numberOfDimension:set.shape[1]-1],axis=1)

def perceptron(set): #single layer
    setAlternating = organizePerceptronSet(np.copy(set))
    x = setAlternating[:,:setAlternating.shape[1]-1]
    y = setAlternating[:,setAlternating.shape[1]-1]
    w = np.zeros(setAlternating.shape[1])
    for k in range(setAlternating.shape[0]):
        xExtended = np.append(1,x[k])
        ycalc = 1 if w.dot(xExtended) > 0 else -1
        w = w + (y[k]-ycalc)*xExtended
    return w
    
def perceptronEstimator(set): #single layer
    w = perceptron(set)
    return lambda x: np.sign(w[0] + sum(wi*x[i] for i,wi in enumerate(w[1:])))
    
def organizePerceptronSet(set):
    for k in range(set.shape[0]-1):
        if set[k][set.shape[1]-1] == set[k+1][set.shape[1]-1]:
            result = searchOrganize(set,k,-set[k][set.shape[1]-1])
            if result != -1:
                set[[k+1,result]] = set[[result,k+1]]
    return set
            
def searchOrganize(set,i,value):
    for k in range(i+2,set.shape[0]-1):
        if set[k][set.shape[1]-1] == value:
            return k
    return -1

'''def quadprogSolve(set,fact=0.1):
    x = set[:,:set.shape[1]-1]
    y = set[:,set.shape[1]-1:]
    G = np.array([[y[i][0]*y[j][0]*x[i].dot(x[j].T) for j in range(set.shape[0])] for i in range(set.shape[0])])
    a = np.ones(set.shape[0])
    C = np.append(y,np.zeros([set.shape[0],2*set.shape[0]]),axis=1)
    for k in range(set.shape[0]):
        C[k][k+1] = 1
        C[k][2*k+1] = -1
    b = np.append(0,np.append(np.zeros(set.shape[0]),fact*np.ones(set.shape[0])))
    return quadprog.solve_qp(G,a,C,b,0)[0]
    
def quadprogSolveCVXOPT(set,fact=0.1):
    x = set[:,:set.shape[1]-1]
    y = set[:,set.shape[1]-1:]
    G = np.array([[y[i][0]*y[j][0]*x[i].dot(x[j].T) for j in range(set.shape[0])] for i in range(set.shape[0])])
    a = -np.ones(set.shape[0])
    C = (np.append(y,np.zeros([set.shape[0],2*set.shape[0]]),axis=1))
    for k in range(set.shape[0]):
        C[k][k+1] = 1
        C[k][2*k+1] = -1
    C = -C.T
    b = -np.append(0,np.append(np.zeros(set.shape[0]),fact*np.ones(set.shape[0])))
    args = [cvxopt.matrix(G),cvxopt.matrix(a),cvxopt.matrix(C),cvxopt.matrix(b)]
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((G.shape[1],))'''

class transfer:
    def __init__(self,sourceSet,destinationSet,vect=[],matrix=[],crossValidationSet=[],shortenDimOfTarget=0,sourceEstimatorPerceptron=False):
        self.sourceSet = sourceSet
        if shortenDimOfTarget <= 0 or shortenDimOfTarget>destinationSet.shape[1]-2:
            self.destinationSet = destinationSet
            self.crossValidationSet = crossValidationSet
        else:
            self.destinationSet = shortenDimension(destinationSet,shortenDimOfTarget)
            if crossValidationSet != []:
                self.crossValidationSet = shortenDimension(crossValidationSet,shortenDimOfTarget)
            else:
                self.crossValidationSet = crossValidationSet
        if not sourceEstimatorPerceptron:
            temp = adaBoost(self.sourceSet)
            temp.runDecisionStump(10)
            self.sourceEstimator = temp.getEstimator()
        else:
            self.sourceEstimator = perceptronEstimator(self.sourceSet)
        self.destinationEstimator = 0
        self.projMatrix = []
        self.projVector = []
        self.matrix = matrix
        self.vect = vect

    def getDestinationEstimator(self):
        return self.destinationEstimator
        
    def crossValidate(self,crossValidationSet):
        return getError(crossValidationSet,self.destinationEstimator)
        
    def __generateXsXt(self):
        ws = perceptron(self.sourceSet)
        xs = np.zeros(self.sourceSet.shape[1]-1)
        for k in range(self.sourceSet.shape[1]-1):
            if ws[k+1] != 0:
                xs = np.append(np.zeros(k),np.append(-ws[0]/ws[k+1],np.zeros(self.sourceSet.shape[1]-2-k)))
                break
        wt = perceptron(self.destinationSet)
        xt = np.zeros(self.destinationSet.shape[1]-1)
        for k in range(self.destinationSet.shape[1]-1):
            if wt[k+1] != 0:
                xt = np.append(np.zeros(k),np.append(-wt[0]/wt[k+1],np.zeros(self.destinationSet.shape[1]-2-k)))
                break
        return xs,xt
        
    def __search(self,xs,xt,benchmark=False):
        if not benchmark:
            currentError,matrix = self.__searchSubRoutine3(xs,xt)
            
            n = 1
            k = 0
            errorPreviousLoop = 0
            
            currentError,matrix = self.__searchSubRoutine1(matrix,xs,xt,currentError)
            errorPreviousLoop = currentError
            
            while currentError>0.1:
                currentError,matrix = self.__searchSubRoutine2(matrix,xs,xt,currentError,n)
                if errorPreviousLoop == currentError:
                    if n == 1:
                        currentError,matrix = self.__searchSubRoutine1(matrix,xs,xt,currentError)
                        if errorPreviousLoop == currentError:
                            n += 1
                    else:
                        n += 1
                        if n == 6:
                            n = 1
                            k = 1
                            while k != 0:
                                if k == 5:
                                    k = 0
                                    currentError,matrix = self.__searchSubRoutine3(xs,xt)
                                else:
                                    matrix *= 2**k
                                    newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                                    if newError<currentError:
                                        currentError = newError
                                        k = 0
                                    else:
                                        matrix /= 4**k
                                        newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                                        if newError<currentError:
                                            currentError = newError
                                            k = 0
                                        else:
                                            k += 1
                else:
                    n = 1
                errorPreviousLoop = currentError
            return matrix
        else:
            count = 1
            currentError,matrix = self.__searchSubRoutine3(xs,xt)
            
            n = 1
            k = 0
            errorPreviousLoop = 0
            
            currentError,matrix = self.__searchSubRoutine1(matrix,xs,xt,currentError)
            errorPreviousLoop = currentError
            
            while currentError>0.1:
                currentError,matrix = self.__searchSubRoutine2(matrix,xs,xt,currentError,n)
                if errorPreviousLoop == currentError:
                    if n == 1:
                        currentError,matrix = self.__searchSubRoutine1(matrix,xs,xt,currentError)
                        if errorPreviousLoop == currentError:
                            n += 1
                    else:
                        n += 1
                        if n == 6:
                            n = 1
                            k = 1
                            while k != 0:
                                if k == 5:
                                    k = 0
                                    count += 1
                                    currentError,matrix = self.__searchSubRoutine3(xs,xt)
                                else:
                                    matrix *= 2**k
                                    newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                                    if newError<currentError:
                                        currentError = newError
                                        k = 0
                                    else:
                                        matrix /= 4**k
                                        newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                                        if newError<currentError:
                                            currentError = newError
                                            k = 0
                                        else:
                                            k += 1
                else:
                    n = 1
                errorPreviousLoop = currentError
            return matrix,count
    
    def __searchSubRoutine1(self,matrix,xs,xt,currentError):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = -matrix[i][j]
                newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                if(newError < currentError):
                    currentError = newError
                else:
                    matrix[i][j] = -matrix[i][j]
        return currentError,matrix
        
    def __searchSubRoutine2(self,matrix,xs,xt,currentError,n):
        newError = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] *= 2**n
                newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                if(newError < currentError):
                    currentError = newError
                else:
                    matrix[i][j] /= 4**n
                    newError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                    if(newError < currentError):
                        currentError = newError
                    else:
                        matrix[i][j] *= 2**n
        return currentError,matrix
        
    def __searchSubRoutine3(self,xs,xt):
        currentError,matrix = 1,0
        while currentError > 0.3:
            proj,matrix,vect = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt)
            currentError = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
        return currentError,matrix
    
    def __generateNearMatrixArray(self,matrix,number,xs,xt):
        Hproj = []
        for k in range(number):
            matrix2 = self.__generateNearMatrix(matrix)
            self.projMatrix.append(matrix2)
            self.projVector.append(xs - matrix2.dot(xt))
            Hproj.append(lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix2,xs,xt): S(P(x)))
        return Hproj
            
    def __generateNearMatrix(self,matrix):
        matrix2 = np.zeros(matrix.shape)
        for i in range(matrix2.shape[0]):
            for j in range(matrix2.shape[1]):
                matrix2[i][j] = matrix[i][j] + (2*np.random.rand()-1)/10 * matrix[i][j]
        return matrix2
        
    def __generateRandomVector(self,n):
        return 10**10*(2*np.random.rand(n)-np.ones(n))

    def __generateRandomMatrix(self,n,p):
        return 2*np.random.rand(n,p)-np.ones([n,p])
        
    def __generateDigitRandomMatrix(self):
        matrix = np.zeros([64,784])
        for l in range(7):
            for k in range(7):
                for i in range(4):
                    for j in range(4):
                        matrix[8*l+k][l*112+4*k+28*i+j] = 0.0625*np.random.rand()
        return matrix

    def __generateProjection(self,n,p,digit=False):
        matrix = []
        if digit:
            matrix = self.__generateDigitRandomMatrix()
        else:
            matrix = self.__generateRandomMatrix(n,p)
        vect = self.__generateRandomVector(n)
        return lambda x,matrix=matrix,vect=vect: vect + matrix.dot(x),matrix,vect
        
    def __generateLinearProjection(self,n,p,xs,xt,digit=False):
        matrix = []
        if digit:
            matrix = self.__generateDigitRandomMatrix()
        else:
            matrix = self.__generateRandomMatrix(n,p)
        vect = xs - matrix.dot(xt)
        return lambda x,matrix=matrix,vect=vect: vect + matrix.dot(x),matrix,vect
        
    def __generateProjectionNoRandomMatrix(self,matrix,xs,xt):
        return lambda x,matrix=matrix,xs=xs,xt=xt: xs - matrix.dot(xt) + matrix.dot(x)

    def __generateProjectionDetermined(self,matrix,vect):
        return lambda x: vect + matrix.dot(x)

    def transfer(self,numberOfTests=1,linear=True,noRandom=True,benchmark=False,noRandomOp1=True,noRandomOp2=True,numberOfBenchmarkTest=100,digit=False):
        if not benchmark:
            self.destinationEstimator,self.projMatrix,self.projVector = 0,[],[]
            if self.matrix != [] and self.vect != []:
                proj = self.__generateProjectionNoRandom(np.array(self.vect),np.array(self.matrix))
                Hproj = lambda x,S=self.sourceEstimator,P=proj: S(P(x))
                self.destinationEstimator = Hproj
            else:
                xs,xt = self.__generateXsXt()
                if self.matrix != []:
                    proj = self.__generateProjectionNoRandomMatrix(np.array(self.matrix),xs,xt)
                    self.destinationEstimator = lambda x,S=self.sourceEstimator,P=proj: S(P(x))
                else:
                    if digit:
                        Hproj = []
                        for k in range(numberOfTests):
                            proj,projMatrix,projVector = 0,0,0
                            if linear:
                                proj,projMatrix,projVector = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt,digit=True)
                            else:
                                proj,projMatrix,projVector = self.__generateProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,digit=True)
                            Hproj.append(lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                            self.projMatrix.append(projMatrix)
                            self.projVector.append(projVector)
                        temp = adaBoost(self.destinationSet)
                        temp.run(np.array(Hproj))
                        self.destinationEstimator = temp.getEstimator()
                    else:
                        if noRandom:
                            matrix,errorLoop = 0,0,1
                            if noRandomOp1:
                                matrix = self.__search(xs,xt)
                            elif noRandomOp1:
                                while errorLoop > 0.1:
                                    proj,matrix,projVector = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt)
                                    errorLoop = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                            else:
                                while errorLoop > 0.1:
                                    proj,matrix,projVector = self.__generateProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt)
                                    errorLoop = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                            Hproj = self.__generateNearMatrixArray(matrix,numberOfTests-1,xs,xt)
                            Hproj.append(lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                            temp = adaBoost(self.destinationSet)
                            temp.run(np.array(Hproj))
                            self.destinationEstimator = temp.getEstimator()
                        else:
                            Hproj = []
                            for k in range(numberOfTests):
                                proj,projMatrix,projVector = 0,0,0
                                if linear:
                                    proj,projMatrix,projVector = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt)
                                else:
                                    proj,projMatrix,projVector = self.__generateProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1)
                                Hproj.append(lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                                self.projMatrix.append(projMatrix)
                                self.projVector.append(projVector)
                            temp = adaBoost(self.destinationSet)
                            temp.run(np.array(Hproj))
                            self.destinationEstimator = temp.getEstimator()
        else:
            standardDeviationMatrixArray,standardDeviationTimeArray,standardDeviationErrorArray,standardDeviationCrossErrorArray = [],[],[],[]
            start = time.time()
            count = 0
            error = 0
            crossError = 0
            for b in range(numberOfBenchmarkTest):
                startBis = time.time()
                countBis = 0
                self.destinationEstimator,self.projMatrix,self.projVector = 0,[],[]
                if self.matrix != [] and self.vect != []:
                    countBis += 1
                    proj = self.__generateProjectionNoRandom(np.array(self.vect),np.array(self.matrix))
                    self.destinationEstimator = lambda x,S=self.sourceEstimator,P=proj: S(P(x))
                else:
                    xs,xt = self.__generateXsXt()
                    if self.matrix != []:
                        countBis += 1
                        proj = self.__generateProjectionNoRandomMatrix(np.array(self.matrix),xs,xt)
                        self.destinationEstimator = lambda x,S=self.sourceEstimator,P=proj: S(P(x))
                    else:
                        if digit:
                            Hproj = []
                            for k in range(numberOfTests):
                                proj,projMatrix,projVector = 0,0,0
                                if linear:
                                    countBis += 1
                                    proj,projMatrix,projVector = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt,digit=True)
                                else:
                                    countBis += 1
                                    proj,projMatrix,projVector = self.__generateProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,digit=True)
                                Hproj.append(lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                                self.projMatrix.append(projMatrix)
                                self.projVector.append(projVector)
                            temp = adaBoost(self.destinationSet)
                            temp.run(np.array(Hproj))
                            self.destinationEstimator = temp.getEstimator()
                        else:
                            if noRandom:
                                matrix,errorLoop = 0,1
                                if noRandomOp1:
                                    matrix,count2 = self.__search(xs,xt,benchmark=True)
                                    countBis += count2
                                elif noRandomOp2:
                                    while errorLoop > 0.1:
                                        countBis += 1
                                        proj,matrix,projVector = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt)
                                        errorLoop = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                                else:
                                    while errorLoop > 0.1:
                                        countBis += 1
                                        proj,matrix,projVector = self.__generateProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1)
                                        errorLoop = getError(self.destinationSet,lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                                countBis += numberOfTests-1
                                Hproj = self.__generateNearMatrixArray(matrix,numberOfTests-1,xs,xt)
                                Hproj.append(lambda x,S=self.sourceEstimator,P=self.__generateProjectionNoRandomMatrix(matrix,xs,xt): S(P(x)))
                                temp = adaBoost(self.destinationSet)
                                temp.run(np.array(Hproj))
                                self.destinationEstimator = temp.getEstimator()
                            else:
                                Hproj = []
                                for k in range(numberOfTests):
                                    proj,projMatrix,projVector = 0,0,0
                                    if linear:
                                        countBis += 1
                                        proj,projMatrix,projVector = self.__generateLinearProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1,xs,xt)
                                    else:
                                        countBis += 1
                                        proj,projMatrix,projVector = self.__generateProjection(self.sourceSet.shape[1]-1,self.destinationSet.shape[1]-1)
                                    Hproj.append(lambda x,S=self.sourceEstimator,P=proj: S(P(x)))
                                    self.projMatrix.append(projMatrix)
                                    self.projVector.append(projVector)
                                temp = adaBoost(self.destinationSet)
                                temp.run(np.array(Hproj))
                                self.destinationEstimator = temp.getEstimator()
                standardDeviationMatrixArray.append(countBis)
                count += countBis
                standardDeviationTimeArray.append(time.time() - startBis)
                temp = getError(self.destinationSet,self.destinationEstimator)
                standardDeviationErrorArray.append(temp)
                error += temp
                if self.crossValidationSet != []:
                    temp = self.crossValidate(self.crossValidationSet)
                    standardDeviationCrossErrorArray.append(temp)
                    crossError += temp
            print("Transfer Learning")
            print("=============================")
            print("=============================")
            mean = error/numberOfBenchmarkTest
            print("Erreur moyenne sur le set de départ:", mean)
            temp = np.array(standardDeviationErrorArray) - mean*np.ones(numberOfBenchmarkTest)
            print("Ecart-type sur l'erreur du set de départ:", (temp.dot(temp.T)/(numberOfBenchmarkTest-1))**0.5)
            print("=============================")
            if self.crossValidationSet != []:
                mean = crossError/numberOfBenchmarkTest
                print("Erreur moyenne sur le set de cross validation:", mean)
                temp = np.array(standardDeviationCrossErrorArray) - mean*np.ones(numberOfBenchmarkTest)
                print("Ecart-type sur l'erreur du set de cross validation:", (temp.dot(temp.T)/(numberOfBenchmarkTest-1))**0.5)
                print("=============================")
            mean = count/numberOfBenchmarkTest
            print("Nombre moyen de matrice générée:", mean)
            temp = np.array(standardDeviationMatrixArray) - mean*np.ones(numberOfBenchmarkTest)
            print("Ecart-type sur le nombre moyen de matrice générée:", (temp.dot(temp.T)/(numberOfBenchmarkTest-1))**0.5)
            print("=============================")
            mean = (time.time() - start)/numberOfBenchmarkTest
            print("Temps moyen pour obtention:", mean)
            temp = np.array(standardDeviationTimeArray) - mean*np.ones(numberOfBenchmarkTest)
            print("Ecart-type sur le temps moyen pour obtention:", (temp.dot(temp.T)/(numberOfBenchmarkTest-1))**0.5)
            
    def test(self,x):
        return self.destinationEstimator(x)