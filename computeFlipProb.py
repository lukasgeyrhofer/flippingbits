#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import sys,math
import itertools

def lesshalftrue(a):
    if np.sum(a) < len(a)/2:
        return True
    else:
        return False

def isflip(a,b):
    return lesshalftrue(a) != lesshalftrue(np.logical_xor(a,b))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-K","--K",default=5,type=int)
    args = parser.parse_args()
    
    flipcount = np.zeros((args.K+1,args.K+1),dtype=int)

    for s0 in range(args.K+1):
        conf = np.repeat([False],args.K+1)
        conf[:s0] = True
        allconf = itertools.permutations(conf)
        for c in allconf:
            for i in range(args.K+1):
                count = 0
                flip = np.repeat([False],args.K+1)
                flip[:i] = True
                allflip = itertools.permutations(flip)
                for fc in allflip:
                    flipcount[s0,i] += isflip(c,fc)
    
    print(flipcount)
                
            


if __name__ == "__main__":
    main()
