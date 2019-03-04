#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import sys,math

def Pxf_hallel(sflip,r = 0.1):
    ret = []
    for s in range(len(sflip)):
        ret.append(r * sflip[s] * np.power(1-r,s))
    return np.vstack(ret).flatten()

def Pxf_lukas(sflip,r = 0.1):
    ret = []
    for s in range(len(sflip)):
        ret.append(r * sflip[s] * np.prod((1-r) + r * (1-sflip[:s])))
    return np.vstack(ret).flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infile",type=str)
    parser.add_argument("-K","--K",type=int,default=5)
    parser.add_argument("-r","--updaterate",type=float,default=.1)
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.infile)
    except:
        raise IOError("could not load file '{}'".format(args.infile))
    
    steps = np.array(data[:,0],dtype=int)
    xflip = data[:,1]
    sflip = data[:,2]
    
    xflipH = Pxf_hallel(sflip, r = args.updaterate)
    xflipL = Pxf_lukas(sflip, r = args.updaterate)

    for s,sf,xf,xfH,xfL in zip(steps,sflip,xflip,xflipH,xflipL):
        print('{:4d} {:14.6e} {:14.6e} {:14.6e} {:14.6e}'.format(s,sf,xf,xfH,xfL))


if __name__ == "__main__":
    main()
