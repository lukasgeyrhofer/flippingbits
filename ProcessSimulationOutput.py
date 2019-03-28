#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles",          default = [],     type = str, nargs = "*")
    parser.add_argument("-o", "--outfileprefix",    default = "norm", type = str)
    parser.add_argument("-v", "--verbose",          default = False,  action = "store_true")
    parser.add_argument("-O", "--OverwriteInPlace", default = False,  action = "store_true")
    parser.add_argument("-C", "--Cumulative",       default = False,  action = "store_true")
    args = parser.parse_args()

    for fn in args.infiles:
        try:    data = np.genfromtxt(fn)
        except: continue

        if args.verbose: print(fn)
        
        n                  = data[:,0]
        Pxfn               = data[:,1]
        Psfn               = data[:,2]
        Pxfgn              = np.zeros(len(n))
        Pxfgn[data[:,6]>0] = data[:,5][data[:,6]>0]/data[:,6][data[:,6]>0]
        Psfgn              = np.zeros(len(n))
        Psfgn[data[:,4]>0] = data[:,3][data[:,4]>0]/data[:,4][data[:,4]>0]
        output             = np.array([n,Pxfn,Psfn,Pxfgn,Psfgn], dtype = np.float).T

        columnheaders      = ['n', 'P[xf,n]', 'P[sf,n]', 'P[xf|n]', 'P[sf|n]']
        
        if np.shape(data)[1] > 7:
            Pxfngsf        = data[:,7]
            Pxfngsf       /= np.sum(Pxfngsf)
            Psfngxf        = data[:,8]
            Psfngxf       /= np.sum(Psfngxf)
            output         = np.concatenate([output,np.array([Pxfngsf,Psfngxf], dtype = np.float).T], axis = 1)

            columnheaders.append('P[xf,n|sf]')
            columnheaders.append('P[sf,n|xf]')
            
        
        if args.Cumulative:
            Cxfn           = np.array([np.sum(Pxf[:m]) for m in range(len(n))])
            Csfn           = np.array([np.sum(Psf[:m]) for m in range(len(n))])
            output         = np.concatenate([output, np.array([Cxfn,Csfn], dtype = np.float).T], axis = 1)
            
            columnheaders.append('P[xf,<n]')
            columnheaders.append('P[sf,<n]')
            
            
        if args.OverwriteInPlace:   outfilename = fn
        else:                       outfilename = args.outfileprefix + '_' + os.path.basename(fn)
        
        np.savetxt(outfilename, output, header = ' '.join(['{:>14s}'.format(h + '  ') for h in columnheaders]), fmt = '%14.6e')


if __name__ == "__main__":
    main()
