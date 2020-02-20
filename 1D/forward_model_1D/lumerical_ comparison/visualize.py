import numpy as np
import matplotlib.pyplot as plt

from fields import nodeField

def plotField(fig_num,Y,Y_lum):
    
    plt.figure(fig_num)
    
    Y_node = nodeField(Y)
    
    plt.plot(np.abs(Y_node),label = 'STSN')
    plt.plot(np.abs(Y_lum),label = 'Lumerical')
    
    plt.ylabel('abs field')
    plt.xlabel('position')
    plt.legend()
        
def graphInput(fig_num,Y,Y_lum):
    
    plotField(fig_num,Y,Y_lum)
        
    plt.title('Input')
    plt.show()
    
def graphOutput(fig_num,Y,Y_lum):
    
    plotField(fig_num,Y,Y_lum)
        
    plt.title('Output')
    plt.show()
    
        