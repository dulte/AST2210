import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    if len(sys.argv)<2:
        print 'Wrong number if input arguments.'
        print 'Usage: python plot_contours.py resultfile.npy'
        sys.exit()
    inputfile = sys.argv[1]
    

    a = np.load(inputfile)
    Q_values = a[0,:]
    n_values = a[1,:]
    lnL = a[2:,:]
    

    plt.hist(n_values,bins=50)
    plt.show()

    resolution = 200.

    q = np.linspace(np.min(Q_values),np.max(Q_values),int(len(Q_values)/resolution))
    n = np.linspace(np.min(n_values),np.max(n_values),int(len(n_values)/resolution))
    
    
    distribution = np.zeros((len(q),len(n)))
    
    for i in range(len(Q_values)):
        q_index = np.argmin(np.abs(q-Q_values[i]))
        n_index = np.argmin(np.abs(n-n_values[i]))
        
        distribution[q_index,n_index] += 1
                    
    distribution /= float(len(Q_values))
    q_dist = np.sum(distribution,axis=0)
    n_dist = np.sum(distribution,axis=1)
    #print distribution
    plt.pcolor(q,n,distribution)
    plt.colorbar()
    plt.show()
    
    plt.contour(q,n,distribution)
    plt.show()
    
    plt.plot(q,q_dist)
    plt.show()
    plt.plot(n,n_dist)
    plt.show()