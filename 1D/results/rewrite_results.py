
import numpy as np
import pickle

def getFile():
	# get file name from user based on weight distribution parameters
	# RETURN_CASE: specifies the return type. 1 - fileName as string and userTime, length and timeChanges as ints
	#										  2 - fileName as string

    userScatter = input("Number of scatterers: ")   # total number of non free space scatterers in masked region
    userTime = input("Time units: ")					# total number of time steps
    userPoints = input("Simulation points: ")			# total number of space steps
    start = input("Starting Weight Index of Mask: ")	# starting index (1 <--> wN)
    end = input("Ending Weight Index of Mask: ")		# ending index (start <--> wN)
    timeChanges = input("Number of time changes: ")		# number of times weight dist. changes
    
    file_id = f"scatter{userScatter}_T{userTime}_N{userPoints}_start{start}_end{end}_tc{timeChanges}" 	#file id

    return file_id
        
def rewrite(file_id):
	# read the trained weight data and return an array of weight values
    
    last_epoch = 15000
    
    epoch_list = np.arange(20,last_epoch + 20,20)
    
    fileFolder = '/Users/travi/Documents/Northwestern/STSN_Paper/code/python/results/' + file_id
    
    for i in epoch_list:
	
        fileName = "/epoch" + str(i) + "_lossAndWeights.p"

        currStatus = pickle.load( open( fileFolder + fileName, "rb" ) )	
        
        print(np.shape(currStatus))
        
        currStatus = np.array(currStatus)

        weights = currStatus[2:]
        
        er = (weights+2)/2
        w = er**-1
        
        currStatus[2:] = w
        
        pickle.dump( currStatus, open( fileFolder + fileName, "wb" ) )
    

    return w

#main
file_id = getFile()

rewrite(file_id)