import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.close('all')

def getFile(userScatter,userTime,userPoints,start,end,timeChanges):
	# get file name from user based on weight distribution parameters
	# RETURN_CASE: specifies the return type. 
   #    1 - fileName as string and userTime, length and timeChanges as ints
	#										  2 - fileName as string

    file_id = f"scatter{userScatter}_T{userTime}_N{userPoints}_start{start}_end{end}_tc{timeChanges}" 	#file id
    
    return file_id, int(userTime), int(timeChanges), start, end

def readWeight(file_id, T, tc, start, end):
	# read the weight data and return an array of weight value

    address = 'C:/Users/travi/Documents/Northwestern/STSN_Paper/code/python/forward_model_1D/weight_data/'
    fileName = address + file_id + '.csv'	
    W = np.genfromtxt(fileName, delimiter = ',')

    if tc == 0:
        W = W[0,:]

    return W

def readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch):
    
    #get file information
    
    file_id, T, tc, start, end = getFile(userScatter,userTime,userPoints,start,end,timeChanges)
    
    #get ground truth weights
    W_gt = readWeight(file_id, T, tc, start, end)
    
    
	# read the trained weight data and return an array of weight values
    
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
    
    
    #get last epoch of weight data
    if tc == 0:
        
        weights = np.repeat(np.reshape(totalData[last_epoch//20-1,2:],(1,-1)),T,axis=0)
    
    else:
        
        L = np.size(totalData[last_epoch//20-1,2:])
        N = L//T
        weights = np.reshape(totalData[last_epoch//20-1,2:],(T,N))
    
    if tc == 0:
        weights = weights[0,int(start)-1:int(end)]
    else:
        weights = weights[:,int(start)-1:int(end)]
        
    return loss, weights, epochs, start, end, W_gt

def percent_error(w_t,w_gt):
    pe = 100*np.abs((w_t**-0.5-w_gt**-0.5)/w_gt**-0.5)
    return pe

def plot_static_weights(fig_num,start,end,weights,pe):
    
    #padding data
    padding = 3
    weights = np.concatenate((np.concatenate((np.ones(padding),weights)),np.ones(padding)))
    pe = np.concatenate((np.concatenate((np.zeros(padding),pe)),np.zeros(padding)))
    start = int(start)-3;end = int(end)+3
    
    fig, ax1 = plt.subplots(1,1,num = fig_num)

    color = 'tab:blue'
    ax1.set_xlabel('position')
    ax1.set_ylabel('n', color=color)
    ax1.bar(np.arange(start,end+1,1), weights**-0.5, 0.5,color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.9,2.1])
    
    ax2 = ax1.twinx()
    
    color = 'tab:red'
    ax2.set_ylabel('percent error', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(start,end+1,1),pe, '-*',color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    
def plot_time_dep_weigths(fig_num,w_gt,w_t,start,end,color_bar):
    
    #padding data
    padding = 3
    w_gt= np.concatenate((np.concatenate((np.ones((60,padding)),w_gt),axis = 1),np.ones((60,padding))),axis = 1)
    w_t= np.concatenate((np.concatenate((np.ones((60,padding)),w_t),axis = 1),np.ones((60,padding))),axis = 1)
    start = int(start)-3;end = int(end)+3
    
    #build figure
    fig, axes = plt.subplots(nrows=2, ncols=1,num = fig_num)
    
#    #set axes ticks/labels
    axes.flat[1].set_xticks(np.arange(0,int(end) - int(start) + 1,1))
#    axes.flat[0].set_yticks(np.arange(9,int(userTime),dt))
#    axes.flat[1].set_yticks(np.arange(9,int(userTime),dt))
    axes.flat[1].set_xticklabels(np.arange(int(start),int(end)+1,1))
#    axes.flat[0].set_yticklabels(time_values)
#    axes.flat[1].set_yticklabels(time_values)

    
    #turn of top x-axis tick marks
    axes.flat[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    #plot data
    im = axes.flat[0].imshow(w_gt**-0.5,aspect='auto',vmin = 1,vmax = 2.18081326672,cmap = 'Blues' )
    im = axes.flat[1].imshow(w_t**-0.5,aspect='auto',vmin = 1,vmax = 2.18081326672,cmap = 'Blues' )
#    print(np.max(w_t**-0.5))
#    print(np.min(w_t**-0.5))
    
    if color_bar ==1:
        #shift and include colobar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    
    #reduce distance between subplots to zero
    fig.subplots_adjust(hspace=0)
    
    plt.show()
    


#-------------------------Read Data-------------------------------------------#
userScatter = '11';userTime = '120';userPoints = '200';start = '100';end = '110';timeChanges = '0';last_epoch = 2100;
    
loss1, weights1, epochs1, start1, end1, W_gt1 = readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch)

userTime = '130';last_epoch = 3000;

loss2, weights2, epochs2, start2, end2, W_gt2 = readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch)

userTime = '140';last_epoch = 6760;

loss3, weights3, epochs3, start3, end3, W_gt3 = readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch)

userScatter = '3';userTime = '60';userPoints = '150';start = '60';end = '62';timeChanges = '1';last_epoch = 5060;
    
loss4, weights4, epochs4, start4, end4, W_gt4 = readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch)

timeChanges = '2';last_epoch = 2800;

loss5, weights5, epochs5, start5, end5, W_gt5 = readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch)

timeChanges = '3';last_epoch = 4720;

loss6, weights6, epochs6, start6, end6, W_gt6 = readTrainedWeight(userScatter,userTime,userPoints,start,end,timeChanges,last_epoch)

#------------------------Calculate Weigth Difference--------------------------#
pe1 = percent_error(W_gt1,weights1)
print('average percent error 1: ' + str(np.average(pe1)))
print('std of percent error 1: ' + str(np.std(pe1)))
pe2 = percent_error(W_gt2,weights2)
print('average percent error 2: ' + str(np.average(pe2)))
print('std of percent error 2: ' + str(np.std(pe2)))
pe3 = percent_error(W_gt3,weights3)
print('average percent error 3: ' + str(np.average(pe3)))
print('std of percent error 3: ' + str(np.std(pe3)))
pe4 = percent_error(W_gt4,weights4)
print('average percent error 4: ' + str(np.average(pe4)))
print('std of percent error 4: ' + str(np.std(pe4)))
pe5 = percent_error(W_gt5,weights5)
print('average percent error 5: ' + str(np.average(pe5)))
print('std of percent error 5: ' + str(np.std(pe5)))
pe6 = percent_error(W_gt6,weights6)
print('total percent error 6: ' + str(np.average(pe6)))
print('std of percent error 6: ' + str(np.std(pe6)))
pe4_5 = percent_error(W_gt4[0:55,:],weights4[0:55,:])
print('average percent error 4 (excluding last 5 time steps): ' + str(np.average(pe4_5)))
print('std of percent error 4 (excluding last 5 time steps): ' + str(np.std(pe4_5)))
pe5_5 = percent_error(W_gt5[0:55,:],weights5[0:55,:])
print('average percent error 5 (excluding last 5 time steps): ' + str(np.average(pe5_5)))
print('std of percent error 5 (excluding last 5 time steps): ' + str(np.std(pe5_5)))
pe6_5 = percent_error(W_gt6[0:55,:],weights6[0:55,:])
print('total percent error 6 (excluding last 5 time steps): ' + str(np.average(pe6_5)))
print('std of percent error 6 (excluding last 5 time steps): ' + str(np.std(pe6_5)))

#--------------------------Plot-----------------------------------------------#
plt.figure(1)
plt.semilogy(epochs1,loss1,label='1')
plt.semilogy(epochs2,loss2,label='2')
plt.semilogy(epochs3,loss3,label='3')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.xlim([0,7000])

plt.figure(2)
plt.semilogy(epochs4,loss4,label='4')
plt.semilogy(epochs5,loss5,label='5')
plt.semilogy(epochs6,loss6,label='6')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.xlim([0,5500])

plot_static_weights(3,start1,end1,weights1,pe1)

plot_static_weights(4,start2,end2,weights2,pe2)

plot_static_weights(5,start3,end3,weights3,pe3)

plot_time_dep_weigths(6,W_gt4,weights4,start4,end4,1)

plot_time_dep_weigths(7,W_gt4,weights4,start4,end4,0)

plot_time_dep_weigths(8,W_gt5,weights5,start5,end5,0)

plot_time_dep_weigths(9,W_gt6,weights6,start6,end6,0)


