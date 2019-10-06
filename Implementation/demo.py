import sets
import matplotlib.pyplot as plt
from TransferClass import *

digitsSourceSet,digitsDestinationSet,digitsCrossValidationSet = sets.generateDigitsSets()
testCas2 = transfer(digitsSourceSet,digitsDestinationSet,crossValidationSet=digitsCrossValidationSet,sourceEstimatorPerceptron=True)
testCas2.transfer(100,noRandom=False)

print(testCas2.test(sets.getData(30009)))
print(testCas2.test(sets.getData(30007)))

fig=plt.figure(1,figsize=(3,3))
fig.add_subplot(1,2,1)
plt.imshow(sets.getImage(30009),cmap=plt.cm.gray_r,interpolation='nearest')
fig.add_subplot(1,2,2)
plt.imshow(sets.getImage(30007),cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()


