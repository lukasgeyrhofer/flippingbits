#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import sys,math
import os

def revert(x):
    return x[::-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infiles",nargs="*",default=[])
    parser.add_argument("-o","--outfileprefix",default="convolve",type=str)
    args = parser.parse_args()


    for fn in args.infiles:
        try:    data = np.genfromtxt(fn)
        except: continue


        n = data[:,0]
        Pxfn    = data[:,1]

        Pxfngsf = data[:,5]
        Psfngxf = data[:,6]
        
        compPxfn = np.array([np.dot(Psfngxf[:m],revert(Pxfngsf[:m])) for m in range(len(n))])

        outfile = args.outfileprefix + '_' + os.path.basename(fn)
        np.savetxt(outfile,np.array([n,Pxfn,compPxfn]).T)

if __name__ == "__main__":
    main()
