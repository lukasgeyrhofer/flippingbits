#!/usr/bin/env python

import numpy as np
import argparse
import sys,math


def LMSQ(x,y):
    n   = len(x)
    sx  = np.sum(x)
    sy  = np.sum(y)
    sxx = np.dot(x,x)
    sxy = np.dot(x,y)
    syy = np.dot(y,y)
    
    denom  = (n*sxx-sx*sx)
    b      = (n*sxy - sx*sy)/denom
    a      = (sy-b*sx)/n
    estim  = np.array([a,b],dtype=np.float)

    sigma2 = syy + n*a*a + b*b*sxx + 2*a*b*sx - 2*a*sy - 2*b*sxy
    cov    = sigma2 / denom * np.array([[sxx,-sx],[-sx,n]],dtype=np.float)

    return estim,cov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles", nargs = "*",default = [])
    parser.add_argument("-m", "--minrange", default = 5, type = float)
    parser.add_argument("-M", "--maxrange", default = 30, type = float)
    args = parser.parse_args()

    for fn in args.infiles:
        try:
            data = np.genfromtxt(fn)
        except:
            continue

        r = float(fn[13:])
        k = int(fn[7:11])
        
        n     = data[:,0] * r
        Pxfgn = data[:,2]
        
        Pxfgn = Pxfgn[n >= args.minrange]
        n = n[n >= args.minrange]
        
        Pxfgn = Pxfgn[n <= args.maxrange]
        n = n[n <= args.maxrange]
        
        n = n[Pxfgn > 0]
        Pxfgn = Pxfgn[Pxfgn > 0]
        
        fit,cov = LMSQ(n,np.log(Pxfgn))
        
        
        print('{:2d} {:.2f} {:14.6e} {:14.6e}'.format(k,r,*fit))
        
        
        


if __name__ == "__main__":
    main()


