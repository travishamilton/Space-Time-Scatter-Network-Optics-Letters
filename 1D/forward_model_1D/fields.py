
import numpy as np

def source(N,start,tc):
    
    wl = 15                         # set wavelength
    FWHM = wl                     # set Full Width at Half Maximum of source
    Np = 3*wl                       # set number of pts in source
    
    c = FWHM/2.35482;               # set variance of guassin wave
    omega = 2*np.pi/wl;                # set angular frequency
    
    if Np > start:
        raise ValueError("Number of source points larger then area before material")
    
    if tc > 0:
        X = np.zeros((3*N,start-Np),dtype = complex)     # intilize input
        
        for i in range(start-Np):
            for j in range(Np):
                #complex guassian wave packet
                X[3*(i + j),i] = np.exp(-1j*omega*j)*np.exp(-(j-np.floor(Np/2))**2/(2*c**2))
            
    else:
        X = np.zeros((3*N,1),dtype = complex)     # intilize input
        
        for j in range(Np):
            #complex guassian wave packet
            X[3*j,0] = np.exp(-1j*omega*j)*np.exp(-(j-np.floor(Np/2))**2/(2*c**2))

    return X

    
def sourceLumerical(Ei):
    
    #grap total number of position points
    N = np.size(Ei)
    #set input to be every third value in source
    X = np.zeros((3*N,1))
    X[::3,:] = np.expand_dims(Ei,axis = 1)
            
    return X

def nodeField(F):
    #Calculates the node field for all samples
    # F: 2D array with shape (feature numbers, sample numbers)
    # F_node: node field with shape(feature numbers/3,sample numbers)
    
    featN,sampN = np.shape(F)               #Get total features and samples of field F
    
    F_forward = np.zeros((featN//3,sampN),dtype = complex)
    F_backward = np.zeros((featN//3,sampN),dtype = complex)
    
    for i in range(sampN):
        F_forward[0:featN//3,i] = F[:,i][0:featN:3]            #Get foward propagating features
        F_backward[0:featN//3,i] = F[:,i][1:featN:3]           #Get backward propagating features
    
    F_node = F_forward+F_backward
    
        
    return F_node

def nodeFields(F):
    #Calculates the node fields for all samples at all times
    # F: 3D array with shape (feature numbers, sample numbers, time steps)
    # F_node: node field with shape(feature numbers/3,sample numbers, time steps)
    
    featN,sampN,T = np.shape(F)
    F_node = np.zeros((featN//3,sampN,T),dtype = complex)
    
    for i in range(T):
        F_node[:,:,i] = nodeField(F[:,:,i])
        
    return F_node

def overlap(F1,F2):
    # calculate the overlap integral over space and time 
    # F1: field of 3D array [feature number, sample number, time steps]
    # F2: field of 3D array [feature number, sample number, time steps]

    #top = np.abs( np.trapz ( np.trapz( F1*np.conj( F2 ) , axis = 0 ) , axis = 1 ) )**2
    top = np.abs( np.trapz( F1*np.conj( F2 ) , axis = 1 ) )**2
    #bottom = np.trapz( np.trapz( np.abs( F1 )**2 , axis = 0 ), axis = 1)*np.trapz( np.trapz( np.abs( F2 )**2 , axis = 0 ), axis = 1 )
    bottom = np.trapz( np.abs( F1 )**2 , axis = 1 )*np.trapz( np.abs( F2 )**2 , axis = 1 )
    overlap = top/bottom
    
    return overlap
    
def writeFields(Fi,Fo,file_id):
    #Writes the input and oupt fields to a csv file by splitting real and 
    #imaginary components into two seperate sample catagories (double samples)
    # Fi: input field with shape (features, samples)
    # Fo: output field with shape (features, samples)
    
    address = 'field_data/'								# file address
    fileName = address + file_id				# name of file including location relative to current file
    
    np.savetxt(fileName + '_out.csv', np.concatenate((np.real(Fo),np.imag(Fo)),axis=1) , delimiter=",")           # save data as csv file
    np.savetxt(fileName + '_in.csv', np.concatenate((np.real(Fi),np.imag(Fi)),axis=1) , delimiter=",")           # save data as csv file
    

    np.concatenate
