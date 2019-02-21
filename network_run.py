#!/usr/bin/env python

import numpy as np
import sys,math
import argparse

import networkclass as nc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N","--NetworkSize",default=100,type = int)
    parser.add_argument("-s","--Steps",default=1000,type=int)
    args = parser.parse_args()

    network = nc.NetworkDynamics(**vars(args))
    
    network.run(args.Steps)
    
    print network.updatehisto

if __name__ == "__main__":
    main()
