import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

fileName = 'scatter11_T120_N200_start100_end110_tc0.csv'

oi_1 = np.genfromtxt(fileName, delimiter = ',')
size_1 = np.size(oi_1)


fileName = 'scatter11_T130_N200_start100_end110_tc0.csv'

oi_2 = np.genfromtxt(fileName, delimiter = ',')
size_2 = np.size(oi_2)


fileName = 'scatter11_T140_N200_start100_end110_tc0.csv'

oi_3 = np.genfromtxt(fileName, delimiter = ',')
size_3 = np.size(oi_3)

plt.figure(1)
plt.semilogy(np.arange(20,size_1*20+20,20),1-oi_1,label='2 unknowns')
plt.semilogy(np.arange(20,size_2*20+20,20),1-oi_2,label='5 unknowns')
plt.semilogy(np.arange(20,size_3*20+20,20),1-oi_3,label='9 unknowns')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.xlim([0,7000])
#plt.xticks([0,4000,8000,12000])

fileName = 'scatter3_T60_N150_start60_end62_tc1.csv'

oi_1 = np.genfromtxt(fileName, delimiter = ',')
size_1 = np.size(oi_1)

fileName = 'scatter3_T60_N150_start60_end62_tc2.csv'

oi_2 = np.genfromtxt(fileName, delimiter = ',')
size_2 = np.size(oi_2)

fileName = 'scatter3_T60_N150_start60_end62_tc3.csv'

oi_3 = np.genfromtxt(fileName, delimiter = ',')
size_3 = np.size(oi_3)

plt.figure(2)
plt.semilogy(np.arange(20,size_1*20+20,20),1-oi_1,label='180 unknowns')
plt.semilogy(np.arange(20,size_2*20+20,20),1-oi_2,label='180 unknowns')
plt.semilogy(np.arange(20,size_3*20+20,20),1-oi_3,label='420 unknowns')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
#plt.xticks([0,2500,5000,7500])
plt.xlim([0,5500])
