#!/usr/bin/env python

import numpy as np
import sys,math
import argparse

import networkclass as nc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-K","--maxK",type=int,default=31)
    parser.add_argument("-S","--steps",type=int,default=100)
    parser.add_argument("-o","--outbasename",type=str,default="out")
    args = parser.parse_args()



    klist = np.arange(start = 3, stop = args.maxK + 1e-6, step = 2,dtype = int)
    steps = np.arange(args.steps,dtype=int)
    
    for k in klist:
        print(k)
        tmpflip = np.zeros(args.steps)
        for s in steps:
            tmpflip[s] = np.sum(nc.ProbGF(s,k,(k+1)/2)[:(k+1)/2])
        np.savetxt(args.outbasename + '_{:03d}'.format(k),np.array([steps,tmpflip]).T)

if __name__ == "__main__":
    main()
            
