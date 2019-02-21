import numpy as np
import sys,math

class NetworkDynamics(object):
    def __init__(self,**kwargs):
        self.__size  = kwargs.get('NetworkSize',100)
        self.__k     = kwargs.get('K',5)
        self.__r     = kwargs.get('r',.1)
        
        self.__nodes       = self.InitializeNodes()
        self.__connections = self.GenerateTopology() * self.GenerateConnectionStrength()
        
        self.__step  = 0
        self.__lastupdate = -np.ones(self.__size,dtype=np.int)
        self.__updatehisto = np.array([])

        
    def step(self):
        update = np.random.choice([True,False], p = [self.__r, 1 - self.__r], size = self.__size)
        tmpNodeCopy = np.copy(self.__nodes)
        for nodeID in np.arange(self.__size)[update]:
            self.__nodes[nodeID] = np.dot(self.__connections[nodeID],np.sign(tmpNodeCopy))
            if np.sign(self.__nodes[nodeID]) != np.sign(tmpNodeCopy[nodeID]):
                self.UpdateHisto(nodeID)
                self.__lastupdate[nodeID] = self.__step
        self.__step += 1


    def run(self,steps = 1):
        for i in range(steps):
            self.step()


    def GenerateTopology(self, indegree = 'deltaK', outdegree = 'binomial', inparams = {}, outparams = {}):
        tmpadj = np.zeros((self.__size,self.__size),dtype=np.int)
        if indegree == 'deltaK' and outdegree == 'binomial':
            for i in range(self.__size):
                k = inparams.get('K',5)
                connections = np.random.choice(self.__size, k, replace = False)
                tmpadj[i][connections] = 1
        else:
            raise NotImplementedError
        
        return tmpadj
    
    
    def GenerateConnectionStrength(self,weightdistr = 'pm1',**kwargs):
        tmpcs = np.zeros((self.__size,self.__size),dtype=np.float)
        if weightdistr == 'pm1':
            tmpcs = np.random.choice([-1,1],size = (self.__size,self.__size))
        else:
            raise NotImplementedError
        return tmpcs
    
    
    def InitializeNodes(self):
        return 2 * np.random.binomial(self.__k,.5,self.__size) - self.__k
    
    
    def UpdateHisto(self,nodeID):
        if self.__lastupdate[nodeID] >= 0:
            if len(self.__updatehisto) <= self.__step - self.__lastupdate[nodeID]:
                self.__updatehisto = np.concatenate([self.__updatehisto,np.zeros(self.__step - self.__lastupdate[nodeID] - len(self.__updatehisto)+1)])
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



