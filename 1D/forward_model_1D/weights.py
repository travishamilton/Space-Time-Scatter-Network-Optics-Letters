import matplotlib.pyplot as plt
import numpy as np
import pickle

def getFile(return_case):
	# get file name from user based on weight distribution parameters
	# RETURN_CASE: specifies the return type. 
   #    1 - fileName as string and userTime, length and timeChanges as ints
	#										  2 - fileName as string

    userScatter = input("Number of scatterers: ")   # total number of non free space scatterers in masked region
    userTime = input("Time units: ")					# total number of time steps
    userPoints = input("Simulation points: ")			# total number of space steps
    start = input("Starting Weight Index of Mask: ")	# starting index (1 <--> wN)
    end = input("Ending Weight Index of Mask: ")		# ending index (start <--> wN)
    timeChanges = input("Number of time changes: ")		# number of times weight dist. changes
    
    L = int(end)-int(start)+1 				# length of trainable region
    
    address = 'weight_data/'								# file address
    file_id = f"scatter{userScatter}_T{userTime}_N{userPoints}_start{start}_end{end}_tc{timeChanges}" 	#file id
    fileName = address + file_id + '.csv'					# name of file including location relative to current file

    if return_case == 1:
        return file_id,fileName, int(userTime), L, int(timeChanges)
    elif return_case == 2:
        return file_id,fileName, int(userPoints), int(start), int(userTime), int(timeChanges),int(end)
    elif return_case == 3:
        return fileName, int(userTime), L, int(timeChanges)
    else:
        raise ValueError("Return case not recognized.")

def readWeight():
	# read the weight data and return an array of weight values

	file_id,fileName,N,start,T,tc,end = getFile(2)								# get fileName from user

	W = np.genfromtxt(fileName, delimiter = ',')

	return file_id,W,N,start,T,tc,end

def readTrainedWeight(file_id,T,tc):
	# read the trained weight data and return an array of weight values
    
    last_epoch = int(input("Last epochs: "))
    
    epoch_list = np.arange(20,last_epoch + 20,20)
    
    table = []
    
    fileFolder = '/Users/travi/Documents/Northwestern/STSN_Paper/code/python/results/' + file_id
    
    for i in epoch_list:
	
        fileName = "/epoch" + str(i) + "_lossAndWeights.p"

        currStatus = pickle.load ( open ( fileFolder + fileName, "rb" ) )	

        table.append( currStatus )
    
    #convert data to np array
    totalData = np.array(table)
    
    #get epoch and loss data
    epochs = totalData[:,0]
    loss = totalData[:,1]
    
    #plot epoch vs. loss
    plt.figure(1000)
    plt.semilogy(epochs,loss)
    plt.xlabel('epochs')
    
    #get last epoch of weight data
    if tc == 0:
        
        weights = np.repeat(np.reshape(totalData[last_epoch//20-1,2:],(1,-1)),T,axis=0)
    
    else:
        
        L = np.size(totalData[last_epoch//20-1,2:])
        N = L//T
        weights = np.reshape(totalData[last_epoch//20-1,2:],(T,N))
    
    plt.figure(1001)												# plot results
    plt.imshow(weights**-0.5, interpolation='nearest')
    plt.colorbar()
    plt.title('Refractive Index - Trained')
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.show()
    
    return totalData, last_epoch

def writeWeight(fig_num):
	# creates a csv file of weights
	# FIG_NUM: figure number to be used to plot weight distribution

    key = input("Produce new weight file(y/n): ")		# ask if user wants to make a new file

    if key == 'y':

        fileName,T,L,tc = getFile(3)					# get fileName from user				

        weight_array = np.ones((T,L))					# intilize weights to one (free space)
        time_index = 0									# start at time zero

        ran_key = input("Produce random distribution of weights(y/n)?: ")
        
        if ran_key == 'n': 
            
            #produce the first tc segments of weights
            for i in range(tc):
                
                #Get time length of current segment
                time_length = int(input(f"Length of time for segment {i+1} : "))				# get length of time for given weight distribution
                if time_length > T-tc:
                    raise ValueError("Time length is too large")
                    
                for j in range(L):
                        
                    # get weight distribution by converting refractive index values given by user
                    weight_array[time_index:
		                time_length+time_index,j] = float(input(f"Refractive index value for position {j+1}: "))**-2

                time_index = time_length+time_index
            
            #produce last segment of weights
            for j in range(L):
                weight_array[time_index:
                    T,j] = float(input(f"Refractive index value for position {j+1}: "))**-2
                
        elif ran_key == 'y':
            
            #produce the first tc segments of weights
            for i in range(tc):
                    
                #Get time length of current segment
                time_length = int(input(f"Length of time for segment {i+1} : "))				# get length of time for given weight distribution
                if time_length > T-tc:
                    raise ValueError("Time length is too large")
                        
                for j in range(L):
                        
                    # get weight distribution by converting refractive index values given by user
                    weight_array[time_index:
		                time_length+time_index,j] = ( np.random.rand(1,L) + 1 )**-2

                time_index = time_length+time_index
            
            #produce last segment of weights
            for j in range(L):
                weight_array[time_index:
                    T,j] = ( np.random.rand(1,L) + 1 )**-2
    
        else:
                raise ValueError("Random Distribution answer not reconginzed")
            
        np.savetxt(fileName, weight_array, delimiter=",")           # save data as csv file
            
    elif key == 'n':
        print("\nNo File Written")
        
    else:
        raise ValueError("Input not recognized. Use y or n.")

def createWeights(W_trained,N,start,fig_num):
    
    T,Lm = np.shape(W_trained)       # get time steps and length of material
    
    Lb = start - 1                  # set length before material
    
    if Lb < 0:
        raise ValueError("Start is less then 1")
       
    La = N-Lm-Lb                    # set length after materail
    if La > N:
        raise ValueError("End lies outside of simulation region")
        
    W_before = np.ones((T,Lb))    # set free space weights before material
    W_after = np.ones((T,La))      # set free space weights after materail
    W = np.concatenate((W_before,W_trained,W_after),1)
    
    plt.figure(fig_num)												# plot results
    plt.imshow(W**-0.5, interpolation='nearest')
    plt.colorbar()
    plt.title('Refractive Index')
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.show()
        
    return W
    