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

    bin = 20

    q_values = np.