#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import sys,math
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infiles",nargs = "*", default = [],type=str)
    parser.add_argument("-o","--outfileprefix",default="cumulative",type=str)
    parser.add_argument("-v","--verbose",default=False,action="store_true")
    args = parser.parse_args()

    for fn in args.infiles:
        if args.verbose: print(fn)
        data = np.genfromtxt(fn)
        
        steps = data[:,0]
        Pxfn  = data[:,1]
        Psfn  = data[:,2]
        
        Pxfgn = np.zeros(len(steps))
        Psfgn = np.zeros(len(steps))
        
        Pxfgn[data[:,4] > 0] = data[:,3][data[:,4] > 0] / data[:,4][data[:,4] > 0]
        Psfgn[data[:,6] > 0] = data[:,5][data[:,6] > 0] / data[:,6][data[:,6] > 0]
        
        xnorm = 1./np.sum(Pxfn)
        snorm = 1./np.sum(Psfn)
        CDxf = np.array([np.sum(Pxfn[:i]) * xnorm for i in range(len(steps))])
        CDsf = np.array([np.sum(Psfn[:i]) * snorm for i in range(len(steps))])
        
        np.savetxt(args.outfileprefix + '_' + fn, np.array([steps,Pxfn,Pxfgn,CDxf,Psfn,Psfgn,CDsf]).T)
        

if __name__ == "__main__":
    main()
