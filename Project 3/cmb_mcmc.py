# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:02:10 2017

@author: dulte
"""

import numpy as np
import scipy.linalg as spl
import sys, os, timeit
import cmb_likelihood_utils as utils
from numba import jit


if __name__ == "__main__":
    if len(sys.argv)<2:
        print 'Wrong number of input arguments.'
        print 'Usage: python cmb_likelihood.py params.py'
        sys.exit()
        
    # Reading parameters from param file into namespace.
    # Tip: this is one possible way of using a parameter file in python, allowing you to keep all your run-specific parameters in a separate file, making it easier to keep track of what values you are using and de-cluttering your code.
    # Disclaimer: This is not an optimal way, just _a_ way. It's unsafe in the sense that there's nothing stopping you from overwriting your parameters within the code, which is bad practice (if they're to be considered constants, at least). Feel free to improve it! 
    namespace={}
    paramfile = sys.argv[1]
    execfile(paramfile,namespace)
    globals().update(namespace)

    runtime_start = timeit.default_timer()

    print 'Loading cmb data from input file %s'%cmbfile
    data = np.load(cmbfile)
    x, y, z, cmb, rms = [data[:,i] for i in range(5)]

    numdata = len(x)
    print 'Number of unmasked pixels to be used for analysis: ', numdata

    print 'Loading beam from file %s, using ells of 0 through %d'%(beamfile,lmax)
    data = np.load(beamfile)
    ells, beam = [data[:,i] for i in range(2)]
    beam = beam[0:lmax+1]

    print 'Loading temperature pixel window from file %s, using ells of 0 through %d'%(pixwinfile,lmax)
    data = np.load(pixwinfile)
    ells, pixwin = [data[:,i] for i in range(2)]
    pixwin = pixwin[0:lmax+1]

    # Finished setup of input data
    # --------------------------------------
    print 'Finished loading data. Now pre-computing noise and foreground covariances'
    N_cov = utils.get_noise_cov(rms)
    F_cov = utils.get_foreground_cov(x,y,z)

    print 'Now pre-computing Legendre polynomials for signal covariance'
    p_ell_ij = utils.get_legendre_mat(lmax,x,y,z)

    time_a = timeit.default_timer()
    print 'Time spent on setup: %f seconds'%(time_a - runtime_start)

    # Finished precomputation
    # ---------------------------------------

    # Setup for the metropolis
    # ---------------------------------------
    
    def calc_lnL(Q,n):
        Cl_model = utils.get_C_ell_model(Q,n,lmax)
        S_cov = utils.get_signal_cov(Cl_model,beam,pixwin,p_ell_ij)
        cov = S_cov + N_cov + F_cov
        return -0.5*utils.get_lnL(cmb,cov)

    lnL = np.zeros(int(metropolis_iterations))
    Q_values = np.zeros_like(lnL)
    n_values = np.zeros_like(lnL)
    Q_values[0] = Q_guess
    n_values[0] = n_guess

    
    lnL[0] = calc_lnL(Q_guess,n_guess)
    
    time_start = timeit.default_timer()

    q_rand = np.random.normal(size=int(metropolis_iterations))
    n_rand = np.random.normal(size=int(metropolis_iterations))

    dice = np.random.uniform(size=int(metropolis_iterations))

    print "Starting Metropolis"
    from progress.bar import Bar
    bar = Bar('Processing', max=int(metropolis_iterations))

 
    for i in range(1,int(metropolis_iterations)):
        if i == 1e2:
            time_end = timeit.default_timer()
            time_to_end = (time_end-time_start)*metropolis_iterations*1e-2 
            print "\nSorry this is going to take %g sec\n" %(time_to_end)
            
        
        new_q = Q_values[i-1] + Q_step*q_rand[i]
        new_n = n_values[i-1] + n_step*n_rand[i]
        
        temp_lnL = calc_lnL(new_q,new_n)   

        diff = temp_lnL-lnL[i-1]
        
        if diff >= 0:
            trow_out_prob = 1
        else:
            trow_out_prob = np.exp(diff)

        # trow_out_prob = np.min(1.,np.exp(potens))
        
        
        if dice[i] <= trow_out_prob:
            lnL[i] = temp_lnL
            Q_values[i] = new_q
            n_values[i] = new_n
        else:
            lnL[i] = lnL[i-1]
            Q_values[i] = Q_values[i-1]
            n_values[i] = n_values[i-1]

        bar.next()

    bar.finish()
    
    # Saving full likelihood in numpy array format. This is faster and easier
    # to read in later, for visualization
    np.save(resultfile,np.vstack([Q_values.T,n_values.T,lnL]))
        
        

        
        
        