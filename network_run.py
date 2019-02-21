#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import sys,math
import argparse

import networkclass as nc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N","--NetworkSize",default=100,type = int)
    parser.add_argument("-s","--Steps",default=1000,type=int)
    parser.add_argument("-n","--reruns", default=20,type=int)
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
        print('{:4d} {:e}'.format(t,h*icount))

if __name__ == "__main__":
    main()
