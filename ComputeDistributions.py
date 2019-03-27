#!/usr/bin/env python

from __future__ import print_function

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
    parser.add_argument("-i","--infiles",nargs = "*", default = [])
    parser.add_argument("-o","--outfileprefix",default="out",type=str)
    args = parser.parse_args()

    for fn in args.infiles:
        try:
            data = np.genfromtxt(fn)
        except:
            continue
        
        

if __name__ == "__main__":
    main()


