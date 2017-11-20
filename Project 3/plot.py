import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == '__main__':
    if len(sys.argv)<2:
        print 'Wrong number if input arguments.'
        print 'Usage: python plot_contours.py resultfile.npy GHz'
        sys.exit()
    inputfile = sys.argv[1]
    band = int(sys.argv[2])
    

    a = np.load(inputfile)
    Q_values = a[0,:]
    n_values = a[1,:]
    lnL = a[2:,:]
    
    print n_values
    

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
    
    print np.sum(q_dist)*dQ, np.sum(n_dist)*dn


    plt.pcolor(q,n,distribution.T)
    plt.colorbar()
    plt.show()
    
    plt.contour(q,n,distribution.T)
    plt.title(r"Distribution of the Posterior $P(Q,n|\mathbf{d})$ for %g GHz" %band)
    plt.xlabel("n")
    plt.ylabel("Q")
    plt.show()
    
    plt.plot(q,q_dist)
    plt.title("Distribution of the values of Q for %g GHz" %band)
    plt.xlabel(r"Q $[\mu K]$")
    plt.ylabel("P(Q)")
    plt.show()
    
    plt.plot(q,savgol_filter(q_dist,7,1))
    plt.title("Distribution of the values of Q with Savitzky-Golay filter for %g GHz" %band)
    plt.xlabel(r"Q $[\mu K]$")
    plt.ylabel("P(Q)")
    plt.show()
    
    
    
    plt.plot(n,n_dist)
    plt.title("Distribution of the values of n for %g GHz" %band)
    plt.xlabel(r"n")
    plt.ylabel("P(n)")
    plt.show()
    
    plt.plot(n,savgol_filter(n_dist,7,1))
    plt.title("Distribution of the values of n with Savitzky-Golay filter for %g GHz" %band)
    plt.xlabel(r"n")
    plt.ylabel("P(n)")
    plt.show()
    
    
    
    ax = plt.subplot(111)
    ax.contour(q,n,distribution.T)

    divider = make_axes_locatable(ax)
    axQ = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax)
    axn = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
    
    axQ.plot(q,q_dist)
    axn.plot(n_dist,n)
    plt.show()
    
    
    
    
    print "n = %g +/- %g" %(np.mean(n_values),np.std(n_values))
    print "Q = %g +/- %g" %(np.mean(Q_values),np.std(Q_values))
