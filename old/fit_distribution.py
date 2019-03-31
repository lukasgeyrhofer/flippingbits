#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import numpy as np
import argparse
import sys,math

from scipy.optimize import curve_fit
from scipy.special import gamma


def LMSQ(x,y):
    n   = len(x)
    sx  = np.sum(x)
    sy  = np.sum(y)
    sxx = np.dot(x,x)
    sxy = np.dot(x,y)
    syy = np.dot(y,y)
    
    denom  = (n*sxx-sx*sx)
    b      = (n*sxy - sx*sy)/denom
    a      = (sy-b*sx)/n
    estim  = np.array([a,b],dtype=np.float)

    sigma2 = syy + n*a*a + b*b*sxx + 2*a*b*sx - 2*a*sy - 2*b*sxy
    cov    = sigma2 / denom * np.array([[sxx,-sx],[-sx,n]],dtype=np.float)

    return estim,cov

def lxexp(x,a,b):
    return np.log(b) + a * np.log(x*b) - b*x - np.log(gamma(a+1.))

def xexp(x,a,b):
    return b/gamma(a+1.) * np.power(x*b,a) * np.exp(-b*x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infiles",nargs="*",default=[])
    parser.add_argument("-m","--maxfev",default=1000,type=int)
    parser.add_argument("-M","--maxlen",default=None,type=int)
    parser.add_argument("-H","--HorizontalOutput",default=False,action="store_true")
    args = parser.parse_args()
    
    for fn in args.infiles:
        
        try:
            data = np.genfromtxt(fn)
            
        except:
            raise IOError
        
        
        if args.maxlen is None:
            steps = data[1:,0]
            distr = data[1:,1]
        else:
            if args.maxlen > 0:
                steps = data[1:args.maxlen,0]
                distr = data[1:args.maxlen,1]
            else:
                idx0  = np.argmax(data[1:,1] == 0)
                steps = data[1:idx0,0]
                distr = data[1:idx0,1]
        
        p0 = [1.,1.]
        
        fit,cov = curve_fit(xexp,steps,distr,p0=p0,maxfev=args.maxfev)
        
        rdistr = distr[distr > 0]
        rsteps = steps[distr > 0]
        
        lfit,lcov = curve_fit(lxexp,rsteps,np.log(rdistr),p0=p0,maxfev=args.maxfev)
        
        if not args.HorizontalOutput:
            print('filename  {}'.format(fn))
            print('normalfit {:14.6e} {:14.6e}'.format(*fit))
            print('logfit    {:14.6e} {:14.6e}'.format(*lfit))
            
            print('f(x)  = {:.6e} * x**{:.6e} * exp(-{:.6e}*x)'.format(np.power(fit[1],fit[0]+1)/gamma(fit[0]+1),fit[0],fit[1]))
            print('lf(x) = {:.6e} * x**{:.6e} * exp(-{:.6e}*x)'.format(np.power(lfit[1],lfit[0]+1)/gamma(lfit[0]+1),lfit[0],lfit[1]))
        
        
        m0 = np.sum(distr)
        m1 = np.dot(steps,distr) / m0
        m2 = np.dot(steps*steps,distr) / m0
        
        im0 = 1./m0
        
        distr_mode = steps[np.argmax(distr)]
        
        distr_median = steps[np.argmin([ (0.5 - im0 * np.sum(distr[:i]))**2 for i in range(len(distr))])]
        
        idx0 = np.argmax(distr == 0)
        if idx0 == 0:
            idx0 = len(distr)
        
        distr_tail = distr[50:idx0]
        steps_tail = steps[50:idx0]
        
        tfit,tcov = LMSQ(steps_tail,np.log(distr_tail))
        
        if not args.HorizontalOutput:
            print('mean      {:14.6e}'.format(m1,np.sqrt(m2 - m1**2)))
            print('variance  {:14.6e}'.format(m2-m1*m1))
            print('mode      {:14.6e}'.format(distr_mode))
            print('median    {:14.6e}'.format(distr_median))
            print('tailslope {:14.6e}'.format(-tfit[1]))
            print('')
        else:
            print('{:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e}'.format(m1,m2-m1**2,distr_mode,distr_median,-tfit[1],fit[0],fit[1],lfit[0],lfit[1]))
    
if __name__ == "__main__":
    main()
    
