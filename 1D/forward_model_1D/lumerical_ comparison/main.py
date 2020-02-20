
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, '/Users/travi/Documents/Northwestern/STSN_Paper/code/python/forward_model_1D')
from layers import trasmitTimeDep
from fields import sourceLumerical,overlap,nodeFields

sys.path.insert(0, '/Users/travi/Documents/Northwestern/STSN_Paper/code/python/forward_model_1D/lumerical_ comparison')
from read import readWeight, readFields
from visualize import graphOutput, graphInput


plt.close('all')

#------------------------------------Setup------------------------------------#							
data_type = np.float32								# set data type of entire model

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

np.random.seed(7)		# seeding the random number generator to reproduce identical results

#-----------------------------------Weights-----------------------------------#
print('---------Read Data----------')
# read weight array W from Lumerical Data
W = readWeight()
# read i/o field array Ei/Eo from Lumerical Data
Ei,Eo,E = readFields()

#--------------------------------Source---------------------------------------#
#convert Ei to STSN source
X = sourceLumerical(Ei)

#---------------------------------Run Model-----------------------------------#
Y,Y_time  = trasmitTimeDep(X, W, 1)

#------------------------------Graph------------------------------------------#
#graph final output
graphOutput(1,Y,Eo)
#graph final input
graphInput(2,X,Ei)

#-----------------------------Calculate Overlap-------------------------------#
ol = overlap(nodeFields(Y_time),E)
print(f"the overlap integral over time and space is: {ol}")
