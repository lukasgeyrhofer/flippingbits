#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import sys,math

from scipy.optimize import curve_fit



def decay(x,a,b,c):
    return c*np.power(0.5 * (1-np.exp(-a*x)),b*x)


def ldecay(x,a,b,c):
    return np.log(c) - b*x * np.log(2.) + b*x*np.log(1-np.exp(-a*x))


def fitdecay():
    p0 = [1,1,.1]
    cutdistr = np.where(sflip[1:] == 0)[0][0]
    csteps = steps[1:cutdistr]
    csflip = sflip[1:cutdistr]

    fit,cov = curve_fit(decay,csteps,csflip,p0 = p0, maxfev = args.maxfev)
    lfit,lcov = curve_fit(ldecay,csteps,np.log(csflip),p0 = p0, maxfev = args.maxfev)

    ret  = "sFlipFit(x) = {} * (0.5 - 0.5 * exp(-{}*x))**({}*x)\n".format(fit[2],fit[0],fit[1])
    ret += "sFlipLFit(x) = {} * (0.5 - 0.5 * exp(-{}*x))**({}*x)\n".format(lfit[2],lfit[0],lfit[1])
    
    return fit,lfit,ret

    
def Pxf_hallel(sflipn,r = 0.1):
    return np.array([r * sflipn[s] * np.power(1-r,s) for s in range(len(sflipn))])


def Pxf_lukas(sflipn,r = 0.1):
    return np.array([r * sflipn[s] * np.prod((1-r) + r * (1-sflipn[:s])) for s in range(len(sflipn))])


def extractP(p):
    return np.array( [p[i]/(1-np.sum(p[:i])) if np.sum(p[:i]) < 1 else 0 for i in range(len(p))], dtype = np.float)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infile",type=str)
    parser.add_argument("-o","--outfile",type=str,default=None)
    parser.add_argument("-K","--K",type=int,default=5)
    parser.add_argument("-r","--updaterate",type=float,default=.1)
    parser.add_argument("-m","--maxfev",type=int,default=1000)
    parser.add_argument("-M","--maxsteps",type=int,default=1000)
    global args
    args = parser.parse_args()


    try:
        data = np.genfromtxt(args.infile,dtype=np.float)
    except:
        raise IOError("could not load file '{}'".format(args.infile))
    
    steps     = np.array(data[:args.maxsteps,0],dtype=int)
    
    Pxfn      = data[:args.maxsteps,1]/np.sum(data[:,1])
    tmpa,tmpb = data[:args.maxsteps,5],data[:args.maxsteps,6]
    Pxfgn     = np.zeros(len(tmpa))
    Pxfgn[tmpb > 0] = tmpa[tmpb > 0]/tmpb[tmpb > 0]
    Pxfgn[0]  = 0
    
    Psfn      = data[:args.maxsteps,2]/np.sum(data[:,2])
    tmpa,tmpb = data[:args.maxsteps,3],data[:args.maxsteps,4]
    Psfgn     = np.zeros(len(tmpa))
    Psfgn[tmpb > 0] = tmpa[tmpb > 0]/tmpb[tmpb > 0]
    Psfgn[0]  = 0
    
    xflipH = Pxf_hallel(Psfgn,r = args.updaterate)
    xflipL = Pxf_lukas(Psfgn,r = args.updaterate)

    xflipE = extractP(Pxfn)
    sflipE = extractP(Psfn)

    if args.outfile is None:    fp = sys.stdout
    else:                       fp = open(args.outfile,'w')
    
    np.savetxt(fp,np.array([steps,Pxfn,Pxfgn,xflipH,xflipL,xflipE,Psfn,Psfgn,sflipE],dtype=np.float).T)
    fp.close()


if __name__ == "__main__":
    main()

