#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import sys,math
import argparse

import networkclass as nc


def main():
    parser = argparse.ArgumentParser()
    parser_io = parser.add_argument_group(description = "==== I/O parameters ====")
    parser_io.add_argument("-o", "--HistoOutfile",         default = 'histo.txt', type = str)
    parser_io.add_argument("-C", "--HistoInputChangeFile", default = None,        type = str)
    parser_io.add_argument("-v", "--verbose",              default = False,       action = "store_true")
    
    parser_net = parser.add_argument_group(description = "==== Network parameters ====")
    parser_net.add_argument("-N", "--NetworkSize",         default = 500,      type = int)
    parser_net.add_argument("-K", "--K",                   default = 5,        type = int)
    parser_net.add_argument("-T", "--InTopologyType",      default = 'deltaK', choices = ['deltaK', 'full'])
    parser_net.add_argument("-r", "--UpdateRate",          default = .1,       type = float)
    
    parser_run = parser.add_argument_group(description = "==== Simulation runs ====")
    parser_run.add_argument("-S", "--Steps",               default = 2000,     type = int)
    parser_run.add_argument("-n", "--reruns",              default = 20,       type = int)
    parser_run.add_argument("-H", "--MaxHistoLength",      default = None,     type = int)
    args = parser.parse_args()

    histoX = list()
    histoS = list()
    
    condprobInput_flip  = np.array([],dtype=np.float)
    condprobInput_total = np.array([],dtype=np.float)
    condprobNodes_flip  = np.array([],dtype=np.float)
    condprobNodes_total = np.array([],dtype=np.float)
    histoatflipNodes    = np.array([],dtype=np.float)
    histoatflipInput    = np.array([],dtype=np.float)
    
    histoinputchanges = list()
    
    for n in range(args.reruns):
        if args.verbose: print('simulating network #{}'.format(n))
        
        # initialize network from scratch and run simulation
        network = nc.NetworkDynamics(**vars(args))
        network.run(args.Steps)
        
        # extract all data from current run
        histoX.append(network.histoX)
        histoS.append(network.histoS)
        
        cpSf,cpSt = network.condprobInput
        if len(condprobInput_flip) < len(cpSf):
            condprobInput_flip  = np.concatenate([condprobInput_flip, np.zeros(len(cpSf) - len(condprobInput_flip))])
            condprobInput_total = np.concatenate([condprobInput_total,np.zeros(len(cpSt) - len(condprobInput_total))])
        condprobInput_flip[:len(cpSf)]  += cpSf
        condprobInput_total[:len(cpSt)] += cpSt
        
        cpXf,cpXt = network.condprobNodes
        if len(condprobNodes_flip) < len(cpXf):
            condprobNodes_flip  = np.concatenate([condprobNodes_flip, np.zeros(len(cpXf) - len(condprobNodes_flip))])
            condprobNodes_total = np.concatenate([condprobNodes_total,np.zeros(len(cpXt) - len(condprobNodes_total))])
        condprobNodes_flip[:len(cpXf)]  += cpXf
        condprobNodes_total[:len(cpXt)] += cpXt
        
        hafX = network.histoatflipnodes
        if len(histoatflipNodes) < len(hafX):
            histoatflipNodes = np.concatenate([histoatflipNodes,np.zeros(len(hafX) - len(histoatflipNodes))])
        histoatflipNodes[:len(hafX)] += hafX
        
        hafS = network.histoatflipinput
        if len(histoatflipInput) < len(hafS):
            histoatflipInput = np.concatenate([histoatflipInput,np.zeros(len(hafS) - len(histoatflipInput))])
        histoatflipInput[:len(hafS)] += hafS
        
        if not args.HistoInputChangeFile is None:
            histoinputchanges.append(network.histoinputchange)
        
        
    # bring all measured histograms to the same size to store them in single file
    histolen = np.max([np.max([len(h) for h in histoX]),np.max([len(h) for h in histoS]),len(condprobInput_total),len(condprobNodes_total),len(histoatflipInput),len(histoatflipNodes)])
    totalhistoX = np.zeros(histolen,dtype = np.float)
    totalhistoS = np.zeros(histolen,dtype = np.float)
    
    if histolen > len(condprobInput_total):
        condprobInput_flip  = np.concatenate([condprobInput_flip,np.zeros(histolen - len(condprobInput_flip))])
        condprobInput_total = np.concatenate([condprobInput_total,np.ones(histolen - len(condprobInput_total))])
    
    if histolen > len(condprobNodes_total):
        condprobNodes_flip  = np.concatenate([condprobNodes_flip,np.zeros(histolen - len(condprobNodes_flip))])
        condprobNodes_total = np.concatenate([condprobNodes_total,np.ones(histolen - len(condprobNodes_total))])
    
    if histolen > len(histoatflipInput):
        histoatflipInput    = np.concatenate([histoatflipInput,np.zeros(histolen - len(histoatflipInput))])
    
    if histolen > len(histoatflipNodes):
        histoatflipNodes    = np.concatenate([histoatflipNodes,np.zeros(histolen - len(histoatflipNodes))])
    
    for h in histoX:
        totalhistoX[:len(h)] += h
    for h in histoS:
        totalhistoS[:len(h)] += h
    
    icountX = 1./np.sum(totalhistoX)
    icountS = 1./np.sum(totalhistoS)
    
    bins = np.arange(histolen)
    
    if args.verbose: print("save histogram recordings to '{}'".format(args.HistoOutfile))
    np.savetxt(args.HistoOutfile,np.array([bins,totalhistoX * icountX, totalhistoS * icountS, condprobInput_flip, condprobInput_total, condprobNodes_flip, condprobNodes_total, histoatflipInput, histoatflipNodes]).T)
    # column:                              1    2                      3                      4                   5                    6                   7                    8                 9

    if not args.HistoInputChangeFile is None:
        l = np.max([len(h) for h in histoinputchanges])
        totalhistoinput = np.zeros(l,dtype=np.int)
        for h in histoinputchanges:
            totalhistoinput[:len(h)] += h
        if args.verbose: print("save input change histogram to '{}'".format(args.HistoInputChangeFile))
        np.savetxt(args.HistoInputChangeFile,np.array([np.arange(l),totalhistoinput],dtype=np.int).T,fmt = '%d')


if __name__ == "__main__":
    main()
