import numpy as np
import matplotlib.pyplot as plt

from fields import nodeField

def plotField(fig_num,Y):
    
    featN,sampN = np.shape(Y)       #get feature and smaple numbers
    Y_node = nodeField(Y)           #get field at node
    
    plt.figure(fig_num)
    
    for i in range(sampN):
        plt.plot(np.abs(Y_node[:,i]))
    
    plt.ylabel('filed amplitude')
    plt.xlabel('position')
        

def graphOutput(fig_num,Y):
    
    plotField(fig_num,Y)
        
    plt.title('Output')
    plt.show()
    
def graphInput(fig_num,Y):
    
    plotField(fig_num,Y)
        
    plt.title('Input')
    plt.show()
        