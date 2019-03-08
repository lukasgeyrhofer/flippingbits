import numpy as np
import sys,math

class NetworkDynamics(object):
    def __init__(self,**kwargs):
        self.__size                 = kwargs.get('NetworkSize',100)
        self.__r                    = kwargs.get('UpdateRate',.1)

        self.__intopology           = dict()
        self.__intopology['K']      = kwargs.get('K',5)
        self.__intopology['type']   = kwargs.get('InTopologyType','deltaK')
        
        self.__outtopology          = dict()
        self.__outtopology['type']  = kwargs.get('OutTopologyType','binomial')
        
        self.__connections          = dict()
        self.__connections['distr'] = kwargs.get('ConnectionDistr','pm1')
        
        self.__nodes                = self.InitializeNodes()
        self.__connections          = self.GenerateTopology() * self.GenerateConnectionStrength()
        
        self.__lastupdate_nodes     = -np.ones(self.__size,dtype=np.int)
        self.__lastupdate_input     = -np.ones(self.__size,dtype=np.int)
        self.__sInputBefore         = np.array(self.__size,dtype=np.float)
        
        self.__maxhistolength       = kwargs.get('MaxHistoLength')
        self.__updatehisto_nodes    = np.array([],dtype=np.int)
        self.__updatehisto_input    = np.array([],dtype=np.int)
        self.__condprobSF_total     = np.array([],dtype=np.float)
        self.__condprobSF_flip      = np.array([],dtype=np.float)
        
        self.__step                 = 0
        self.__verbose              = kwargs.get("verbose",False)

        
    def step(self):
        # get vector of nodes to update
        update               = np.random.choice([True,False], p = [self.__r, 1 - self.__r], size = self.__size)
        
        # copy nodes and compute full dynamics
        tmpNodeCopy          = np.copy(self.__nodes)
        sInput               = np.dot(self.__connections,np.sign(tmpNodeCopy))
        
        # update nodes with probability r
        self.__nodes[update] = sInput[update]
        
        # check, where changes occurred
        # record histograms for node and input flip times
        updateNodesHisto     = np.where(np.sign(self.__nodes) != np.sign(tmpNodeCopy),True,False)
        updateInputHisto     = np.where(np.sign(sInput) != np.sign(self.__sInputBefore),True,False)
        
        xupdates = self.UpdateXHisto(updateNodesHisto,update)
        supdates = self.UpdateSHisto(updateInputHisto)

        self.UpdateCondProbFlip(updateInputHisto)
        
        self.__lastupdate_input[updateInputHisto] = self.__step
        self.__lastupdate_nodes[updateNodesHisto] = self.__step



        # output
        if self.__verbose:
            print('{:5d} {:3d} {:3d} ['.format(self.__step,xupdates,supdates) + ' '.join(['{:3d}'.format(x) for x in self.__nodes]) + ']')

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
        elif self.__intopology['type'] == 'full':
            tmpadj = np.ones((self.__size,self.__size))
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
    
    
    def CheckMaxHistoLength(self,steps):
        if self.__maxhistolength is None:
            return True
        else:
            if steps <= self.__maxhistolength:
                return True
            else:
                return False
    
    def UpdateXHisto(self,updateNodesHisto,update):
        countupdates = 0
        for nodeID in np.arange(self.__size)[updateNodesHisto]:
            if self.__lastupdate_nodes[nodeID] >= 0 and update[nodeID] and self.CheckMaxHistoLength(self.__step - self.__lastupdate_nodes[nodeID]):
                if len(self.__updatehisto_nodes) <= self.__step - self.__lastupdate_nodes[nodeID]:
                    self.__updatehisto_nodes = np.concatenate([self.__updatehisto_nodes,np.zeros(self.__step - self.__lastupdate_nodes[nodeID] - len(self.__updatehisto_nodes)+1,dtype=np.int)])
        
                self.__updatehisto_nodes[self.__step - self.__lastupdate_nodes[nodeID]] += 1
                countupdates += 1
        return countupdates
    
    
    def UpdateSHisto(self,updateInputHisto):
        countupdates = 0
        for nodeID in np.arange(self.__size)[updateInputHisto]:
            if self.__lastupdate_input[nodeID] >= 0 and self.CheckMaxHistoLength(self.__step - self.__lastupdate_input[nodeID]):
                if len(self.__updatehisto_input) <= self.__step - self.__lastupdate_input[nodeID]:
                    self.__updatehisto_input = np.concatenate([self.__updatehisto_input,np.zeros(self.__step - self.__lastupdate_input[nodeID] - len(self.__updatehisto_input)+1,dtype=np.int)])
                
                self.__updatehisto_input[self.__step - self.__lastupdate_input[nodeID]] += 1
                countupdates += 1
        return countupdates
    
    
    def UpdateCondProbFlip(self,updateInputHisto):
        maxtime = np.max(self.__step - self.__lastupdate_input)
        if np.any(self.__lastupdate_input == -1):
            maxtime -= 1
        if maxtime >= len(self.__condprobSF_total) and self.CheckMaxHistoLength(maxtime):
            self.__condprobSF_total = np.concatenate([self.__condprobSF_total,np.zeros(1)])
            self.__condprobSF_flip  = np.concatenate([self.__condprobSF_flip, np.zeros(1)])
        
        for i in range(self.__size):
            if self.__lastupdate_input[i] >= 0 and self.CheckMaxHistoLength(self.__step - self.__lastupdate_input[i]):
                self.__condprobSF_total[self.__step - self.__lastupdate_input[i]] += 1
                if updateInputHisto[i]:
                    self.__condprobSF_flip[self.__step - self.__lastupdate_input[i]] += 1
    
    
    def __getattr__(self,key):
        if key == 'histoX':
            return self.__updatehisto_nodes
        if key == 'histoS':
            return self.__updatehisto_input
        elif key == 'nodes':
            return self.__nodes
        elif key == 'connections':
            return self.__connections
        elif key == 'condprobSF':
            return self.__condprobSF_flip/self.__condprobSF_total
        elif key == 'condprobSF_counts':
            return self.__condprobSF_flip,self.__condprobSF_total


    def __getitem__(self,key):
        return self.__nodes[key]




## ===========================================
##  analytical and approximative expressions
## ===========================================

# evaluate generating function with SymPy (symbolic python)
import sympy
from scipy.special import gamma


def genfunc(x,y,z,a0 = 5,b0 = 0):
    return (x * sympy.cosh(z) + y * sympy.sinh(z))**a0 * (x * sympy.sinh(z) + y * sympy.cosh(z))**b0

def ProbGF(t = 0, k = 5, a0 = 0):
    invfactorial = np.array([1./gamma(i+1) for i in range(k+1)],dtype = np.float)
    x,y,z = sympy.symbols('x y z')
    expr = genfunc(x,y,z,a0,k - a0)
    expr = sympy.diff(expr,z,t).subs(z,0.)
    r = np.ones(k+1) / (k ** t)
    for i in range(k+1):
        tmp = sympy.diff(expr,x,i).evalf(subs = {x : 0.})
        tmp = sympy.diff(tmp,y,k-i).evalf(subs = {y : 0.})
        r[i] *= tmp * invfactorial[i] * invfactorial[k-i]
    return r


# approximations based on mean field analysis using 'Ehrenfest Urn'
def Psflip(step, r = .1, K = 5):
    return 0.5*(1.-np.exp(-2.*r*step/K))

def Pxflip(step, r = .1, K = 5):
    if isinstance(step,(list,tuple,np.ndarray)):
        return np.array([r * Psflip(s, r ,K) * np.prod([1-r+r*(1-Psflip(i, r, K)) for i in np.arange(1,s)]) for s in step])
    else:
        return r * Psflip(step, r ,K) * np.prod([1-r+r*(1-Psflip(i, r, K)) for i in np.arange(1,step)])
