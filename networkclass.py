import numpy as np
import sys,math

class NetworkDynamics(object):
    def __init__(self,**kwargs):
        self.__size                 = kwargs.get('NetworkSize',100)
        self.__r                    = kwargs.get('UpdateRate',.1)

        self.__intopology           = dict()
        self.__intopology['K']      = kwargs.get('K',5)
        self.__intopology['type']   = kwargs.get('InTopType','deltaK')
        
        self.__outtopology          = dict()
        self.__outtopology['type']  = kwargs.get('OutTopType','binomial')
        
        self.__connections          = dict()
        self.__connections['distr'] = kwargs.get('ConnectionDistr','pm1')
        
        self.__nodes                = self.InitializeNodes()
        self.__connections          = self.GenerateTopology() * self.GenerateConnectionStrength()
        
        self.__step                 = 0
        self.__lastupdate_nodes     = -np.ones(self.__size,dtype=np.int)
        self.__lastupdate_input     = -np.ones(self.__size,dtype=np.int)
        self.__updatehisto_nodes    = np.array([],dtype=np.int)
        self.__updatehisto_input    = np.array([],dtype=np.int)
        self.__sInputBefore         = np.array(self.__size,dtype=np.float)

        
    def step(self):
        # get vector of nodes to update
        update               = np.random.choice([True,False], p = [self.__r, 1 - self.__r], size = self.__size)
        
        # copy nodes and compute full dynamics
        tmpNodeCopy          = np.copy(self.__nodes)
        sInput               = np.dot(self.__connections,np.sign(tmpNodeCopy))
        
        # update nodes with probability r
        self.__nodes[update] = sInput[update]
        
        print(self.__nodes)
        
        # check, where changes occurred
        # record histograms for node and input flip times
        updateNodesHisto     = np.where(np.sign(self.__nodes) == np.sign(tmpNodeCopy),True,False)
        updateInputHisto     = np.where(np.sign(sInput) == np.sign(self.__sInputBefore),True,False)
        
        self.UpdateXHisto(updateNodesHisto)
        self.UpdateSHisto(updateInputHisto)
        
        self.__lastupdate_input[updateInputHisto] = self.__step
        self.__lastupdate_nodes[updateNodesHisto] = self.__step
        
        # prepare for next step
        self.__sInputBefore  = sInput
        self.__step         += 1


    def run(self,steps = 1):
        for i in range(steps):
            self.step()


    def GenerateTopology(self):
        tmpadj = np.zeros((self.__size,self.__size),dtype=np.int)
        if self.__intopology['type'] == 'deltaK' and self.__outtopology['type'] == 'binomial':
            for i in range(self.__size):
                connections = np.random.choice(self.__size, self.__intopology.get('K',5), replace = False)
                tmpadj[i][connections] = 1
        else:
            raise NotImplementedError
        
        return tmpadj
    
    
    def GenerateConnectionStrength(self):
        tmpcs = np.zeros((self.__size,self.__size),dtype=np.float)
        if self.__connections['distr'] == 'pm1':
            tmpcs = np.random.choice([-1,1],size = (self.__size,self.__size))
        else:
            raise NotImplementedError
        return tmpcs
    
    
    def InitializeNodes(self):
        return 2 * np.random.binomial(self.__intopology['K'],.5,self.__size) - self.__intopology['K']
    
    
    def UpdateXHisto(self,updateNodesHisto):
        for nodeID in np.arange(self.__size)[updateNodesHisto]:
            if self.__lastupdate_nodes[nodeID] >= 0:
                if len(self.__updatehisto_nodes) <= self.__step - self.__lastupdate_nodes[nodeID]:
                    self.__updatehisto_nodes = np.concatenate([self.__updatehisto_nodes,np.zeros(self.__step - self.__lastupdate_nodes[nodeID] - len(self.__updatehisto_nodes)+1,dtype=np.int)])
            
                self.__updatehisto_nodes[self.__step - self.__lastupdate_nodes[nodeID]] += 1
    
    
    def UpdateSHisto(self,updateInputHisto):
        for nodeID in np.arange(self.__size)[updateInputHisto]:
            if self.__lastupdate_input[nodeID] >= 0:
                if len(self.__updatehisto_input) <= self.__step - self.__lastupdate_input[nodeID]:
                    self.__updatehisto_input = np.concatenate([self.__updatehisto_input,np.zeros(self.__step - self.__lastupdate_input[nodeID] - len(self.__updatehisto_input)+1,dtype=np.int)])
                
                self.__updatehisto_input[self.__step - self.__lastupdate_input[nodeID]] += 1
        
    
    
    def __getattr__(self,key):
        if key == 'updatehisto':
            return self.__updatehisto_nodes
        elif key == 'nodes':
            return self.__nodes
        elif key == 'connections':
            return self.__connections


    def __getitem__(self,key):
        return self.__nodes[key]



