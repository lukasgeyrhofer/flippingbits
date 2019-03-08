#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import sys,math
import argparse

import networkclass as nc


def main():
    parser = argparse.ArgumentParser()
    parser_io = parser.add_argument_group(description = "==== I/O parameters ====")
    parser_io.add_argument("-o", "--HistoOutfile",    default = 'histo.txt', type = str)
    parser_io.add_argument("-v", "--verbose",         default = False,       action = "store_true")
    
    parser_net = parser.add_argument_group(description = "==== Network parameters ====")
    parser_net.add_argument("-N", "--NetworkSize",    default = 100,      type = int)
    parser_net.add_argument("-K", "--K",              default = 5,        type = int)
    parser_net.add_argument("-T", "--InTopologyType", default = 'deltaK', choices = ['deltaK', 'full'])
    parser_net.add_argument("-r", "--UpdateRate",     default = .1,       type = float)
    
    parser_run = parser.add_argument_group(description = "==== Simulation runs ====")
    parser_run.add_argument("-S", "--Steps",          default = 1000,     type = int)
    parser_run.add_argument("-n", "--reruns",         default = 20,       type = int)
    parser_run.add_argument("-H", "--MaxHistoLength", default = 2000,     type = int)
    args = parser.parse_args()

    histoX = list()
    histoS = list()
    
    condprobSF_flip  = np.array([],dtype=np.float)
    condprobSF_total = np.array([],dtype=np.float)
    
    for n in range(args.reruns):
        if args.verbose:
            print('simulating network #{}'.format(n))
        network = nc.NetworkDynamics(**vars(args))
        network.run(args.Steps)
        histoX.append(network.histoX)
        histoS.append(network.histoS)
        
        cpf,cpt = network.condprobSF_counts
        if len(condprobSF_flip) < len(cpf):
            condprobSF_flip  = np.concatenate([condprobSF_flip, np.zeros(len(cpf) - len(condprobSF_flip))])
            condprobSF_total = np.concatenate([condprobSF_total,np.zeros(len(cpt) - len(condprobSF_total))])
        
        condprobSF_flip[:len(cpf)]  += cpf
        condprobSF_total[:len(cpt)] += cpt
        
    
    histolen = np.max([np.max([len(h) for h in histoX]),np.max([len(h) for h in histoS]),len(condprobSF_total)])
    totalhistoX = np.zeros(histolen,dtype = np.int)
    totalhistoS = np.zeros(histolen,dtype = np.int)
    
    if histolen > len(condprobSF_total):
        condprobSF_flip  = np.concatenate([condprobSF_flip,np.zeros(histolen - len(condprobSF_flip))])
        condprobSF_total = np.concatenate([condprobSF_total,np.ones(histolen - len(condprobSF_total))])
    
    for h in histoX:
        totalhistoX[:len(h)] += h
    for h in histoS:
        totalhistoS[:len(h)] += h
    
    icountX = 1./np.sum(totalhistoX)
    icountS = 1./np.sum(totalhistoS)
    
    bins = np.arange(histolen)
    
    if args.verbose:
        print("save histogram recordings to '{}'".format(args.HistoOutfile))
    np.savetxt(args.HistoOutfile,np.array([bins,totalhistoX * icountX, totalhistoS * icountS, condprobSF_flip, condprobSF_total]).T)

if __name__ == "__main__":
    main()
