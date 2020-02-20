
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from fields import overlap,nodeFields

data_type = np.float32

# ---------------------- Propagation Matrix ---------------------#
def translate(X):
	feat_N, Np = np.shape(X)	# get shape of the data, O(1) constant run-time
	
	T = np.zeros((feat_N,feat_N),dtype = data_type)

	for j in range(feat_N//3):
	    if j>=1:
	    	T[3*j - 2,3*j]=1
	    if j<=feat_N//3-2:
	    	T[3*j + 3,3*j + 1]=1
	    T[3*j+2,3*j+2] = 1

	return T

# ---------------------- Scatter Matrix Generation ----------------------#
def scatterSubmatrices(w):

    N = np.size(w)	# extract number of weights, N
    
    A = [ [1,0,0] , [0,1,0] , [0,0,-2] ]				# initialize A matrix
    A2 = A
    B = [ [-1,0,2] , [0,-1,2] , [0,0,1] ]				# initialize B matrix
    B2 = B

    # diagonlize A and B N times
    for _ in range(N-1):
        A = sp.block_diag((A,A2))
        B = sp.block_diag((B,B2))

    U_left = [ [-1,-1] , [1,0] , [0,1] ]				# initialize U_left matrix
    U_left_2 = U_left
    U_right = [ [1] , [1] , [1] ]						# initialize U_right matrix
    U_right_2 = U_right

    # diagonlize U N times
    for _ in range(N-1):
        U_left = sp.block_diag((U_left,U_left_2))
        U_right = sp.block_diag((U_right,U_right_2))
	
    U = sp.hstack((U_left,U_right))
    U_inv = sp.linalg.inv(U.tocsc())
    
    return U.tocsc(),U_inv.tocsc(),A.tocsc(),B.tocsc()

def scatter(w,U,U_inv,A,B):
    
    N = np.size(w)	# extract number of weights, N
        
    # create the weight matrix through the eigenvalue matrix V and eigenvector matrix U
    V = np.zeros(2*N)
    V = np.concatenate( ( V , np.multiply( 3.0, w ) ) , 0 )
    V = sp.diags(V)

    W = U @ V @ U_inv

    S = W @ A + B	# create scatter matrix for 1 time unit

    return S

def trasmitTimeDep(X,W,return_type):
    #transmits field X through strucutre W for t steps
    #
    # INPUT
    #
    # X: field array containing feature numbers by sample numbers worth of data
    # W: weight array contaning time steps by position steps worth of data
    # t: number of time steps
    # RETURN_TYPE: returns Y for 0 and Y and Y_time for 1
    #
    # OUTPUT
    #
    # Y: output field containing same shape as X
    # Y_TIME: output field with third dimensions for output at each time step

    featN,sampN = np.shape(X)
    t,N = np.shape(W)
    
	#Create translation matrix
    T = translate(X)
    
    #build submatrices of scatter matrix
    U,U_inv,A,B = scatterSubmatrices(W[0,:])
    
    #Record field at final time step only
    Y = np.zeros((featN,sampN),dtype = complex)
    
    #Record field at different time steps
    Y_time = np.zeros((featN,sampN,t),dtype = complex)
		
    for i in range(t):
        curr_w = W[i,:]
        curr_w = np.reshape( curr_w, [-1] )
        S = scatter(curr_w,U,U_inv,A,B)
        TS = T @ S
        
        if i is 0:
            tmp = TS
        else:
            tmp = tmp @ TS
            
        if return_type is 1:
            Y_time[:,:,i] = tmp @ X
        
    Y = tmp @ X 
    
    #Change Y_time to node values
    if return_type == 0:
        return Y
    elif return_type == 1:
        return Y, Y_time

def transmit(x,W,N):
	# N is now a vector of indicating units of time per weight in w
	# W is now a 2-d matrix of weights, each row of W corresponding to the set of weights for each time unit

	#Create propagation matrix
    P0 = np.transpose(propagate(x))
    S = np.transpose(scatter(W))
    SP0 = np.matmul(S,P0)
    tmp = SP0
    
    for i in range(N-1):
        tmp = np.matmul(tmp,SP0)
        
    out = np.matmul(x,tmp)

    return out

def transmitTrained(last_epoch,totalData,X,T,Y_time,file_id,tc,end):
    
    oi = np.zeros(last_epoch//20)
    
    epoch_list = np.arange(20,last_epoch+20,20)
    
    for i in range(last_epoch//20):
        
        if tc == 0:
            
            W_trained = np.repeat(np.reshape(totalData[i,2:],(1,-1)),T,axis=0)
        
        else:
        
            L = np.size(totalData[last_epoch//20-1,2:])
            N = L//T
            W_trained = np.reshape(totalData[i,2:],(T,N))
            
        Y_trained, Y_time_trained  = trasmitTimeDep(X, W_trained, 1)
        
        
        
        #------------------------------Convert to node fields-----------------#
        F_trained = nodeFields(Y_time_trained)
        F = nodeFields(Y_time)
        
        #------------------------------Overlap Integral-----------------------#
        oi[i] = np.average(overlap(F_trained[end,:,:],F[end,:,:]))
        
    plt.figure(10000)
    plt.semilogy(epoch_list,1-oi)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Space-Time Overlap')
    
    
    plt.figure(10001)
    plt.plot(np.abs(F_trained[end,0,:]),label='trained')
    plt.plot(np.abs(F[end,0,:]),label='ground truth')
    plt.xlabel('Position')
    plt.ylabel('Mag')
    plt.title('Fields')
    plt.legend()

    
    address = 'overlap_data/'
    fileName = address+file_id+'.csv'
    np.savetxt(fileName, oi, delimiter=",")

