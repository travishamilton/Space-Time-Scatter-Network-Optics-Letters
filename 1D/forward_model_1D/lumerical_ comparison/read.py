
import numpy as np

def readWeight():
	# read the weight data given by Lumerical and return a 2D array of weight values

    address = 'lumerical_data/'								# file address
    file_id = 'Lumerical_Weights_1D.csv' 	#file id
    fileName = address + file_id						# name of file including location relative to current file
    
    W = np.genfromtxt(fileName, delimiter = ',')
    
    return W

def readFields():
    # read i/o field data given by Lumerical and return a 1D array of field values

    address = 'lumerical_data/'								# file address
    
    file_id = 'Lumerical_Input_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    Ei = np.genfromtxt(fileName_input, delimiter = ',')
    
    file_id = 'Lumerical_Output_1D.csv' 	#file id
    fileName_output = address + file_id						# name of file including location relative to current file
    
    Eo = np.genfromtxt(fileName_output, delimiter = ',')
    
    file_id = 'Lumerical_Space_Time_1D.csv' 	#file id
    fileName_output = address + file_id						# name of file including location relative to current file
    
    E = np.genfromtxt(fileName_output, delimiter = ',')
    
    E = np.expand_dims(E,axis = 1)
    
    return Ei,Eo,E


    