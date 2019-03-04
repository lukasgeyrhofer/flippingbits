#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import sys,math
import argparse

import networkclass as nc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-K","--maxK",type=int,default=10)
    parser.add_argument("-S","--steps",type=int,default=30)
    parser.add_argument("-o","--outbasename",type=str,default="out")
    args = parser.parse_args()

    klist = np.arange(start = 3, stop = args.maxK + 1e-6, step = 2, dtype = int)
    steps = np.arange(args.steps, dtype = int)
    
    for k in klist:
        
        print(k)
        
        isflipped = np.ones(args.steps + 1)
        flipprob  = np.zeros(args.steps)
        
        for s in steps:
            flipprob[s]   = np.sum(nc.ProbGF(s, k, (k+1)/2)[:(k+1)/2])
        
            if flipprob[s] > 1 or flipprob[s] < 0:
                flipprob[s] = 0
                break
        
        print(flipprob)
        
        
        
        for s in steps:
            isflipped[s]  = np.prod(1. - flipprob[:s])
            isflipped[s] *= flipprob[s]
        
        np.savetxt(args.outbasename + '_{:03d}'.format(k),np.array([steps,flipprob,isflipped[:s+1]]).T)


if __name__ == "__main__":
    main()
            
