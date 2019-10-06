import matplotlib.pyplot as plt

#graphe 1
x = [k for k in range(10,31)]

time1 = [117.815,81.198,133.195,71.707,29.240,34.576,25.565,28.098,26.895,27.644,27.328,27.767,13.317,20.192,3.088,3.085,3.086,3.092,3.077,3.073,3.080]
time2 = [5.843,5.895,5.766,5.493,4.694,5.442,5.767,5.449,4.838,5.077,5.655,5.354,4.626,7.096,3.157,3.172,3.187,3.202,3.238,3.305,3.324]

matriceGen1 = [295582,200369,338458,166144,61766,80846,60269,66483,63504,65635,64913,66036,27510,45735,354,334,343,344,311,300,312]
matriceGen2 = [119,118,116,114,109,112,112,109,108,109,110,110,106,119,100,100,100,100,100,100,100]

fig,ax1 = plt.subplots()
ax1.set_xlabel('Number of dimensions kept on the target set',fontsize='30')
ax1.plot(x,time1,'bo',label='Time in s (A random)')
ax1.plot(x,time2,'ro',label='Time in s (with the pseudo-random gradient descent algorithm)')
ax1.set_ylabel('Time (in s)',fontsize='20')
plt.legend(bbox_to_anchor=(0.8,1))

ax2 = ax1.twinx()
ax2.plot(x,matriceGen1,'b*',label='Number of random matrix generated (A random)')
ax2.plot(x,matriceGen2,'r*',label='Number of random matrix generated (with the pseudo-random gradient descent algorithm)')
ax2.set_ylabel('Number of random matrix generated',fontsize='20')
plt.legend(bbox_to_anchor=(0.8,0.9))

plt.xlim(30,10) 
plt.show()


#graphe 2
x = [k for k in range(10,31)]

deltaMinMax = [0.02814,1.3891,2.7136,9.2016,223.773,0.019887,0.092404,0.1278,0.02801,0.04559,0.008519,20.47,23.62,156.39,2217.8,0.10006,0.62766,0.9608,0.291,0.3872,0.08078]

matriceGen = [295582,200369,338458,166144,61766,80846,60269,66483,63504,65635,64913,66036,27510,45735,354,334,343,344,311,300,312]

fig,ax1 = plt.subplots()
ax1.set_xlabel('Number of dimensions kept on the target set',fontsize='30')
ax1.plot(x,deltaMinMax,'ro',label='Difference between the 2 extrema for this dimension in target set')
ax1.set_ylabel('Difference between the 2 extrema (dim n) in target set',color='red',fontsize='20')
plt.legend(bbox_to_anchor=(0.35,1))

ax2 = ax1.twinx()
ax2.plot(x,matriceGen,'b*',label='Number of random matrix generated (A random)')
ax2.set_ylabel('Number of random matrix generated',color='blue',fontsize='20')
plt.legend(bbox_to_anchor=(0.8,0.95))

plt.xlim(30,10) 
plt.show()