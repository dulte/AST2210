import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import savgol_filter

#plt.style.use('seaborn')

def get_dist(inputfile,band):
    a = np.load(inputfile)
    Q_values = a[0,:]
    n_values = a[1,:]
    lnL = a[2:,:]

    

    plt.hist(n_values,bins=50)
    plt.show()

    resolution = 250.
#    resolution = 500


    q = np.linspace(np.min(Q_values),np.max(Q_values),int(len(Q_values)/resolution))
    n = np.linspace(np.min(n_values),np.max(n_values),int(len(n_values)/resolution))

    
    
    distribution = np.zeros((len(q),len(n)))
    
    for i in range(len(Q_values)):
        q_index = np.argmin(np.abs(q-Q_values[i]))
        n_index = np.argmin(np.abs(n-n_values[i]))
        
        
        distribution[q_index,n_index] += 1
                    
    dQ = q[1] - q[0]
    dn = n[1] - n[0]
                    
    distribution /= float(len(Q_values)*dn*dQ)
    

    
    q_integral = np.sum(distribution,axis=1)*dQ
    n_integral = np.sum(distribution,axis=0)*dn
    
    q_dist = q_integral/(np.sum(q_integral)*dQ)
    n_dist = n_integral/(np.sum(n_integral)*dn)
    
    print "#############################"
    print "Data for %g GHz" %band
    print "n = %g +/- %g" %(np.mean(n_values),np.std(n_values))
    print "Q = %g +/- %g" %(np.mean(Q_values),np.std(Q_values))
    print "#############################"
    
    return distribution, q_dist, n_dist, q, n




if __name__ == '__main__':
    if len(sys.argv)<2:
        print 'Wrong number if input arguments.'
        print 'Usage: python plot_contours.py resultfile53.npy resultfile90.npy'
        sys.exit()
    inputfile53 = sys.argv[1]
    inputfile90 = sys.argv[2]

    distribution53, q_dist53, n_dist53, q53, n53 = get_dist(inputfile53,53)
    distribution90, q_dist90, n_dist90, q90, n90 = get_dist(inputfile90,90)
    
    plt.plot(q53,q_dist53,label="53 GHz")
    plt.plot(q90,q_dist90,label="90 GHz")
    plt.title("Distribution of the values of Q")
    plt.xlabel(r"Q $[\mu K]$")
    plt.ylabel("P(Q)")
    plt.legend()
    plt.show()
    

    plt.plot(q53,savgol_filter(q_dist53,7,1),label="53 GHz")
    plt.plot(q90,savgol_filter(q_dist90,7,1),label="90 GHz")
    plt.title("Distribution of the values of Q with Savitzky-Golay filter")
    plt.xlabel(r"Q $[\mu K]$")
    plt.ylabel("P(Q)")
    plt.legend()
    plt.show()
    
    
    
    plt.plot(n53,n_dist53,label="53 GHz")
    plt.plot(n90,n_dist90,label="90 GHz")
    plt.title("Distribution of the values of n")
    plt.xlabel(r"n")
    plt.ylabel("P(n)")
    plt.legend()
    plt.show()
    

    plt.plot(n53,savgol_filter(n_dist53,7,1),label="53 GHz")
    plt.plot(n90,savgol_filter(n_dist90,7,1),label="90 GHz")
    plt.title("Distribution of the values of n with Savitzky-Golay filter")
    plt.xlabel(r"n")
    plt.ylabel("P(n)")
    plt.legend()
    plt.show()
    

