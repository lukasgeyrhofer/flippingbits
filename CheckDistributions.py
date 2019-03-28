#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import argparse
import sys,math
from os.path import basename

def revert(x):
    return x[::-1]




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles",       default = [],           type = str, nargs = "*",)
    parser.add_argument("-o", "--outfileprefix", default = "CheckDistr", type = str)
    parser.add_argument("-F", "--fftoutfiles",   default = "CDfft",      type = str)
    parser.add_argument("-m", "--maxtime",       default = None,         type = int)
    parser.add_argument("-v", "--verbose",       default = False,        action = "store_true")
    parser.add_argument("-f", "--fourthorder",   default = False,        action = "store_true")
    args = parser.parse_args()

    for fn in args.infiles:
        if args.verbose:    print(fn)
        try:                data = np.genfromtxt(fn)
        except:             continue
        
        n       = data[:,0]
        if not args.maxtime is None:
            maxtime = args.maxtime
        else:
            maxtime = len(n)
        n       = n[:maxtime]
        Pxfn    = data[:maxtime,1] # P[xf,n]
        Psfn    = data[:maxtime,2] # P[sf,n]
        Pxfgn   = data[:maxtime,3] # P[xf|n]
        Psfgn   = data[:maxtime,4] # P[sf|n]
        Pxfngsf = data[:maxtime,5] # P[xf,n|sf]
        Psfngxf = data[:maxtime,6] # P[sf,n|xf]

        PxfnCOMP0 = np.array([np.dot(Psfngxf[:m],revert(Pxfngsf[:m])) for m in range(len(n))])
        PxfnCOMP2 = np.zeros(maxtime)
        for m in range(maxtime):
            for k in range(maxtime-m):
                for l in range(maxtime-m-k):
                    PxfnCOMP2[m] += Psfngxf[m] * Psfn[k] * Psfn[l] * Pxfngsf[maxtime-m-k-l-1]
        
        PxfnCOMP4 = np.copy(PxfnCOMP2)
        if args.fourthorder:
            for m in range(maxtime):
                for k in range(maxtime-m):
                    for l in range(maxtime-m-k):
                        for i in range(maxtime-m-k-l):
                            for j in range(maxtime-m-k-l-i):
                                PxfnCOMP4[m] += Psfngxf[m] * Psfn[k] * Psfn[l] * Psfn[i] * Psfn[j] * Pxfngsf[maxtime-m-k-l-i-j-1]
        
        Fxfz    = np.fft.fft(Pxfn)
        Fsfz    = np.fft.fft(Psfn)
        Fxfgz   = np.fft.fft(Pxfgn)
        Fsfgz   = np.fft.fft(Psfgn)
        Fxfzgsf = np.fft.fft(Pxfngsf)
        Fsfzgxf = np.fft.fft(Psfngxf)
        
        FxfzCOMP0 = Fxfzgsf * Fsfzgxf
        FxfzCOMP2 = Fxfzgsf * Fsfzgxf * (1 + Fsfz**2)
        FxfzCOMP4 = Fxfzgsf * Fsfzgxf * (1 + Fsfz**2 + Fsfz**4)
        
        fPxfn0 = np.real(np.fft.ifft(FxfzCOMP0))
        fPxfn2 = np.real(np.fft.ifft(FxfzCOMP2))
        fPxfn4 = np.real(np.fft.ifft(FxfzCOMP4))
        
        FxfzAT = Fxfz**2
        FsfzAT = Fsfz**2
        
        fPxfnAT = np.fft.ifft(FxfzAT)
        fPsfnAT = np.fft.ifft(FsfzAT)
        
        outfilename = args.outfileprefix + '_' + basename(fn) 
        np.savetxt(outfilename, np.array([n, Pxfn, Psfn, Pxfgn, Psfgn, Pxfngsf, Psfngxf, FxfzCOMP0, FxfzCOMP2, FxfzCOMP4, fPxfn0, fPxfn2, fPxfn4, fPxfnAT, fPsfnAT]).T)
        
        
        fftoutfile = args.fftoutfiles + '_' + basename(fn)
        z = np.fft.fftfreq(maxtime)
        np.savetxt(fftoutfile,np.array([z,Fxfz,Fsfz,FxfzCOMP0,FxfzCOMP2,FxfzCOMP4]).T)
        
        
                                          
if __name__ == "__main__":                   
    main()                                   


