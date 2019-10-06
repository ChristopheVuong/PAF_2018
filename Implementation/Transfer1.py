import sets
from TransferClass import *

'''digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets()

testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
testCas2.transfer(10,noRandom=False,benchmark=True,numberOfBenchmarkTest=100,digit=True)'''

'''#brenchmark array 1
for i in range(10):
    for j in range(i+1,10):
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("Avec la fonction digit pour les nombres ",i," et ",j,":")
        print("=============================")
        print("=============================")
        print("=============================")
        digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets(num1=i,num2=j)
        testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
        testCas2.transfer(100,noRandom=False,benchmark=True,numberOfBenchmarkTest=10,digit=True)

#brenchmark array 2
for i in range(10):
    for j in range(i+1,10):
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("En total random pour les nombres ",i," et ",j,":")
        print("=============================")
        print("=============================")
        print("=============================")
        digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets(num1=i,num2=j)
        testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
        testCas2.transfer(100,noRandom=False,benchmark=True,numberOfBenchmarkTest=10)
        
#brenchmark array 3
for i in range(10):
    for j in range(i+1,10):
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("=============================")
        print("Avec l'algorithme pseudo-random de descente par gradient pour les nombres ",i," et ",j,":")
        print("=============================")
        print("=============================")
        print("=============================")
        digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets(num1=i,num2=j)
        testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
        testCas2.transfer(100,noRandom=True,noRandomOp1=True,noRandomOp2=True,benchmark=True,numberOfBenchmarkTest=10)'''

#benchmark array 4
i = 1
j = 8
print("=============================")
print("=============================")
print("=============================")
print("=============================")
print("=============================")
print("=============================")
print("Avec la fonction digit pour les nombres ",i," et ",j,":")
print("=============================")
print("=============================")
print("=============================")
digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets(num1=i,num2=j)
testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
testCas2.transfer(100,noRandom=False,benchmark=True,numberOfBenchmarkTest=10,digit=True)
        
print("=============================")
print("=============================")
print("=============================")
print("=============================")
print("=============================")
print("=============================")
print("En total random pour les nombres ",i," et ",j,":")
print("=============================")
print("=============================")
print("=============================")
digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets(num1=i,num2=j)
testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
testCas2.transfer(100,noRandom=False,benchmark=True,numberOfBenchmarkTest=10)