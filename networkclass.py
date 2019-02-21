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
        self.__lastupdate           = -np.ones(self.__size,dtype=np.int)
        self.__updatehisto          = np.array([],dtype=np.int)

        
    def step(self):
        # get vector of nodes to update
        update = np.random.choice([True,False], p = [self.__r, 1 - self.__r], size = self.__size)
        
        tmpNodeCopy = np.copy(self.__nodes)
        
        for nodeID in np.arange(self.__size)[update]:
            # xi = \sum_j wij sign(xj)
            self.__nodes[nodeID] = np.dot(self.__connections[nodeID],np.sign(tmpNodeCopy))
            # record event if xi flips sign
            if np.sign(self.__nodes[nodeID]) != np.sign(tmpNodeCopy[nodeID]):
                self.UpdateHisto(nodeID)
                self.__lastupdate[nodeID] = self.__step
        self.__step += 1


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
    
    
    def UpdateHisto(self,nodeID):
        if self.__lastupdate[nodeID] >= 0:
            if len(self.__updatehisto) <= self.__step - self.__lastupdate[nodeID]:
                self.__updatehisto = np.concatenate([self.__updatehisto,np.zeros(self.__step - self.__lastupdate[nodeID] - len(self.__updatehisto)+1,dtype=np.int)])
            self.__updatehisto[self.__step - self.__lastupdate[nodeID]] += 1
    
    
    def __getattr__(self,key):
        if key == 'updatehisto':
            return self.__updatehisto
        elif key == 'nodes':
            return self.__nodes
        elif key == 'connections':
            return self.__connections


    def __getitem__(self,key):
        return self.__nodes[key]



