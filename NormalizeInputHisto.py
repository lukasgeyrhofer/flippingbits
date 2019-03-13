#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import argparse
import sys,math

from scipy.special import gamma,gammaln
from scipy.optimize import curve_fit

def binom(n,K,r):
    return gamma(K+1)/(gamma(n+1) * gamma(K-n+1)) * np.power(r/K,n) * np.power(1.-r/K,K-n)

def lbinom(n,K,r):
    return gammaln(K+1) - gammaln(n+1) - gammaln(K-n+1) + n*np.log(r/K) + (K-n)*np.log(1.-r/K)

def lbinomNOK(n,r):
    return gammaln(Kglobal+1) - gammaln(n+1) - gammaln(Kglobal-n+1) + n*np.log(r/Kglobal) + (Kglobal-n)*np.log(1.-r/Kglobal)

def lbinom2(n,K,p):
    return gammaln(K+1) - gammaln(n+1) - gammaln(K-n+1) + n*np.log(p) + (K-n)*np.log(1.-p)
    

def ExtractKr(filename):
    try:
        K = int(filename.split('.')[np.where([x[:1] == 'K' for x in filename.split('.')])[0][0]][1:])
        ir = np.where([x[:1] == 'r' for x in filename.split('.')])[0][0]+1
        r = float(filename.split('.')[ir]) * 10**-len(filename.split('.')[ir])
        return K,r
    except:
        return None,None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles",       default = [],     nargs = "*")
    parser.add_argument("-o", "--outfileprefix", default = "norm", type = str)
    parser.add_argument("-M", "--maxfev",        default = 1000,   type = int)
    args = parser.parse_args()
    
    global Kglobal
    if len(args.infiles) > 0:
        for fn in args.infiles:
            data = np.genfromtxt(fn,dtype=np.float)
            K,r = ExtractKr(fn)
            Kglobal = K * 1.
            
            norm = np.sum(data[:,1]) * 1.

            n         = np.arange(K+1)
            measured  = data[:,1]/np.sum(data[:,1])
            if len(measured) < K+1:
                measured = np.concatenate([measured,np.zeros(K + 1 - len(measured))])
            predicted = binom(n,K,r)
            
            fit,cov   = curve_fit(binom,n,measured,p0 = [K,r],maxfev = args.maxfev)
            fitted    = binom(n,fit[0],fit[1])
            
            lfit,lcov = curve_fit(lbinom,n[measured>0],np.log(measured[measured>0]), p0 = [K,r], maxfev = args.maxfev)
            lfitted   = binom(n,lfit[0],lfit[1])
            
            kfit,kcov = curve_fit(lbinomNOK,n[measured>0],np.log(measured[measured>0]), p0 = [r], maxfev = args.maxfev)
            kfitted   = binom(n,K,kfit[0])
            
            pfit,pcov = curve_fit(lbinom2,n[measured>0],np.log(measured[measured>0]), p0 = [K,r/K], maxfev = args.maxfev)
            pfitted   = binom(n,pfit[0],pfit[1])
            
            print("{:20s} {:d} {:.2f} {:.6f} {:.6f} {:.6f} {:.6f} {:6f} {:6f} {:6f}".format(fn,K,r,fit[0],fit[1],lfit[0],lfit[1],kfit[0],pfit[0],pfit[1]))
        
            np.savetxt(args.outfileprefix + '_' + fn,np.array([n,measured,predicted,fitted,lfitted,kfitted,pfitted],dtype=np.float).T)
    
if __name__ == "__main__":
    main()
    
