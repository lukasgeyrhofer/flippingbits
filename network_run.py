#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import sys,math
import argparse

import networkclass as nc


def Psflip(step, r = .1, K = 5):
    return 0.5*(1.-np.exp(-2.*r*step/K))

def Pxflip(step, r = .1, K = 5):
    if isinstance(step,(list,tuple,np.ndarray)):
        return np.array([r * Psflip(s, r ,K) * np.prod([1-r+r*(1-Psflip(i, r, K)) for i in np.arange(1,s)]) for s in step])
    else:
        return r * Psflip(step, r ,K) * np.prod([1-r+r*(1-Psflip(i, r, K)) for i in np.arange(1,step)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N","--NetworkSize",default=100,type = int)
    parser.add_argument("-s","--Steps",default=1000,type=int)
    parser.add_argument("-n","--reruns", default=20,type=int)
    parser.add_argument("-r","--UpdateRate", default = .1, type=float)
    parser.add_argument("-K","--K", default = 5, type = int)
    args = parser.parse_args()

    histo = list()

    for n in range(args.reruns):
        #print('{}'.format(n))
        network = nc.NetworkDynamics(**vars(args))
        network.run(args.Steps)
        histo.append(network.updatehisto)
    
    l = np.max([len(h) for h in histo])
    totalhisto = np.zeros(l,dtype = np.int)
    for h in histo:
        totalhisto[:len(h)] += h
    
    icount = 1./np.sum(totalhisto)
    for t,h in enumerate(totalhisto):
        print('{:4d} {:e} {:e}'.format(t,h*icount),Pxflip(t,args.UpdateRate,args.K))

if __name__ == "__main__":
    main()
