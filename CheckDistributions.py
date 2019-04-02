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
    parser.add_argument("-i", "--infiles",           default = [],              type = str, nargs = "*",)
    parser.add_argument("-o", "--outfileprefix",     default = "CheckDistr",    type = str)
    parser.add_argument("-F", "--fftoutfiles",       default = "CheckDistrFFT", type = str)
    parser.add_argument("-M", "--maxorderFFT",       default = 4,               type = int)
    parser.add_argument("-m", "--maxtime",           default = None,            type = int)
    parser.add_argument("-v", "--verbose",           default = False,           action = "store_true")
    parser.add_argument("-C", "--DirectConvolution", default = False,           action = "store_true")
    args = parser.parse_args()

    for fn in args.infiles:
        if args.verbose:    print(fn)
        try:                data = np.genfromtxt(fn)
        except:             continue
        
        n       = data[:,0]
        if not args.maxtime is None:    maxtime = np.min([args.maxtime,len(n)])
        else:                           maxtime = len(n)
            
        n       = n[:maxtime]
        Pxfn    = data[:maxtime,1] # P[xf,n]
        Psfn    = data[:maxtime,2] # P[sf,n]
        Pxfgn   = data[:maxtime,3] # P[xf|n]
        Psfgn   = data[:maxtime,4] # P[sf|n]
        Pxfngsf = data[:maxtime,5] # P[xf,n|sf]
        Psfngxf = data[:maxtime,6] # P[sf,n|xf]

        PxfnCOMP0 = np.array([np.dot(Psfngxf[:m],revert(Pxfngsf[:m])) for m in range(maxtime)])
        
        PxfnCOMP2 = np.zeros(maxtime)
        if args.DirectConvolution:
            for m in range(maxtime):
                for k in range(maxtime-m):
                    for l in range(maxtime-m-k):
                        for i in range(maxtime-m-k-l):
                            PxfnCOMP2[m] += Psfngxf[k] * Psfn[l] * Psfn[i] * Pxfngsf[m-k-l-i-1]
        
        
        PxfnCOMP4 = np.copy(PxfnCOMP2)
        if args.DirectConvolution:
            for m in range(maxtime):
                for k in range(maxtime-m):
                    for l in range(maxtime-m-k):
                        for i in range(maxtime-m-k-l):
                            for j in range(maxtime-m-k-l-i):
                                for r in range(maxtime-m-k-l-i-j):
                                    PxfnCOMP4[m] += Psfngxf[k] * Psfn[l] * Psfn[i] * Psfn[j] * Psfn[r] * Pxfngsf[m-k-l-i-j-r-1]
        
        
        PxfnAT    = np.array([np.dot(Pxfn[:m],revert(Pxfn[:m])) for m in range(maxtime)])
        PsfnAT    = np.array([np.dot(Psfn[:m],revert(Psfn[:m])) for m in range(maxtime)])
        
        
        # use FFT to compute same distributions
        z         = np.real(np.fft.fftfreq(maxtime))
        Fxfz      = np.fft.fft(Pxfn)
        Fsfz      = np.fft.fft(Psfn)
        Fxfgz     = np.fft.fft(Pxfgn)
        Fsfgz     = np.fft.fft(Psfgn)
        Fxfzgsf   = np.fft.fft(Pxfngsf)
        Fsfzgxf   = np.fft.fft(Psfngxf)
        
        FxfzCOMP0 = Fxfzgsf * Fsfzgxf

        FxfzAT    = Fxfz**2
        FsfzAT    = Fsfz**2
        
        fPxfnAT   = np.fft.ifft(FxfzAT)
        fPsfnAT   = np.fft.ifft(FsfzAT)
        
        
        # generate output matrices
        output    = np.array([n, Pxfn, Psfn, Pxfgn, Psfgn, Pxfngsf, Psfngxf, np.real(fPxfnAT), np.real(fPsfnAT), PxfnAT, PsfnAT, PxfnCOMP0, PxfnCOMP2, PxfnCOMP4], dtype = np.float).T
        #                     1  2     3     4      5      6        7        8                 9                 10      11      12         13         14

        fftoutput = np.array([z, np.real(Fxfz), np.imag(Fxfz), np.real(Fsfz), np.imag(Fsfz)]).T
        #                     1  2              3              4              5

        FxfzCOMP  = dict()
        fPxfnCOMP = dict()
        for order in np.arange(start = 0, stop = args.maxorderFFT + 1e-2, step = 2, dtype = np.int):
            FxfzCOMP[order]  = FxfzCOMP0 * np.sum([np.power(Fsfz,2*a) for a in range(order/2)], axis = 0)
            fPxfnCOMP[order] = np.fft.ifft(FxfzCOMP[order])
            
            output           = np.concatenate([output,    np.array( [np.real(fPxfnCOMP[order]), np.imag(fPxfnCOMP[order])] ).T], axis = 1)
            fftoutput        = np.concatenate([fftoutput, np.array( [np.real(FxfzCOMP[order]), np.imag(FxfzCOMP[order])] ).T], axis = 1)


        # write output to files
        outfilename = args.outfileprefix + '_' + basename(fn)
        np.savetxt(outfilename, output, fmt = '%14.6e')
        
        fftoutfile = args.fftoutfiles + '_' + basename(fn)
        np.savetxt(fftoutfile,fftoutput, fmt = '%14.6e')
        
                                          
if __name__ == "__main__":                   
    main()                                   


