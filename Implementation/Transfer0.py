import sets
from TransferClass import *

'''#brenchmark array 1
for k in range(21):
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("Avec ",30-k," dimensions en arrivée:")
    print("=============================")
    print("=============================")
    print("=============================")
    testCas1 = transfer(sets.sourceSet,sets.destinationSet,crossValidationSet=sets.crossValidationSet,shortenDimOfTarget=30-k,sourceEstimatorPerceptron=True)
    testCas1.transfer(100,noRandom=True,noRandomOp1=False,noRandomOp2=True,benchmark=True,numberOfBenchmarkTest=100)'''

'''#brenchmark array 2
min,max = getMinMax(sets.destinationSet)
w = np.delete(perceptron(sets.destinationSet),0)
print(np.multiply(np.array(max)-np.array(min),w))'''

'''#brenchmark array 3
for k in range(21):
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("Avec ",30-k," dimensions en arrivée:")
    print("=============================")
    print("=============================")
    print("=============================")
    testCas1 = transfer(sets.sourceSet,sets.destinationSet,crossValidationSet=sets.crossValidationSet,shortenDimOfTarget=30-k,sourceEstimatorPerceptron=True)
    testCas1.transfer(100,noRandom=True,noRandomOp1=True,noRandomOp2=True,benchmark=True,numberOfBenchmarkTest=100)'''
    
'''#brenchmark array 4
for k in range(5,6):
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("Avec ",10*k," samples dans le set de départ:")
    print("=============================")
    print("=============================")
    print("=============================")
    testCas1 = transfer(sets.sourceSet,reduceSample(sets.destinationSet,10*k),crossValidationSet=sets.crossValidationSet,sourceEstimatorPerceptron=True)
    testCas1.transfer(100,noRandom=True,noRandomOp1=False,noRandomOp2=True,benchmark=True,numberOfBenchmarkTest=100)'''
    
'''#brenchmark array 5
print("Without Transfer Learning")
for k in range(7):
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("Avec ",10*k," samples dans le set de départ:")
    print("=============================")
    print("=============================")
    print("=============================")
    destinationSet = reduceSample(sets.destinationSet,10*k)
    i,error,start,estimator = 0,1,0,0
    while error>0.1:
        i += 1
        start = time.time()
        estimator = adaBoost(destinationSet)
        estimator.runDecisionStump(i)
        error = estimator.getError()
    print("Nombre de decision stumps par dimension du target set:", i)
    print("Erreur sur le set de départ:", error)
    print("Erreur sur le set de cross validation:", getError(sets.crossValidationSet,estimator.getEstimator()))
    print("Temps pour obtention:", (time.time() - start))'''
    
'''#brenchmark array 6
print("Without Transfer Learning")
for k in range(21):
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("Avec ",30-k," dimensions en arrivée:")
    print("=============================")
    print("=============================")
    print("=============================")
    destinationSet,crossValidationSet = shortenDimension(sets.destinationSet,30-k),shortenDimension(sets.crossValidationSet,30-k)
    i,error,start,estimator = 0,1,0,0
    while error>0.1:
        print(i)
        i += 1
        start = time.time()
        estimator = adaBoost(destinationSet)
        estimator.runDecisionStump(i)
        error = estimator.getError()
    print("Nombre de decision stumps par dimension du target set:", i)
    print("Erreur sur le set de départ:", error)
    print("Erreur sur le set de cross validation:", getError(crossValidationSet,estimator.getEstimator()))
    print("Temps pour obtention:", (time.time() - start))'''