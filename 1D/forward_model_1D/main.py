
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, '/Users/travi/Documents/Northwestern/STSN_Paper/code/python/forward_model_1D')

from weights import writeWeight, readWeight, createWeights, readTrainedWeight
from layers import trasmitTimeDep,transmitTrained
from fields import source, writeFields, nodeFields, nodeField
from visualize import graphOutput, graphInput

plt.close('all')

#------------------------------------Setup------------------------------------#							
data_type = np.float32								# set data type of entire model

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

np.random.seed(7)		# seeding the random number generator to reproduce identical results


################################# Ground Truth ################################
###############################################################################

#----------------------------------Write Data---------------------------------#
print('---------------------------------------')
print('Write Data')
print('---------------------------------------')
writeWeight(1)

#-----------------------------------Weights-----------------------------------#
print('---------------------------------------')
print('Read Data')
print('---------------------------------------')
file_id,W,N,start,T,tc,end = readWeight()
# create weight array W
W = createWeights(W,N,start,2)

#--------------------------------Source---------------------------------------#
X = source(N, start,tc)

#---------------------------------Run Model-----------------------------------#
print('---------------------------------------')
print('Run Model')
print('---------------------------------------')


Y, Y_time  = trasmitTimeDep(X, W, 1)

#------------------------------Graph------------------------------------------#
#graph final output
graphOutput(3,Y)
#graph input
graphInput(4,X)

#----------------------------SaveData-----------------------------------------#
writeFields(X,Y,file_id)


################################# Trained #####################################
###############################################################################

key_trained = input("Run model with trained data (y/n)?: ")	

if key_trained == 'y':

    #-------------------------Load Trained Weights----------------------------#
    print('---------------------------------------')
    print('Read Trained Weights')
    print('---------------------------------------')
    totalData,last_epoch = readTrainedWeight(file_id,T,tc)
    
    #-------------------------------Run Model---------------------------------#
    print('---------------------------------------')
    print('Run Model W/ Trained Weights')
    print('---------------------------------------')
    transmitTrained(last_epoch,totalData,X,T,Y_time,file_id,tc,end)
    
        
else:
    
    print('---------------------------------------')
    print('Model not executed on trained data.')
    print('---------------------------------------')