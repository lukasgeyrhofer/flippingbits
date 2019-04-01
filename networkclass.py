import numpy as np
import sys,math

class NetworkDynamics(object):
    def __init__(self,**kwargs):
        self.__size                 = kwargs.get('NetworkSize',500)
        self.__r                    = kwargs.get('UpdateRate',.1)

        self.__intopology           = dict()
        self.__intopology['K']      = kwargs.get('K',5)
        self.__intopology['type']   = kwargs.get('InTopologyType','deltaK')
        
        self.__outtopology          = dict()
        self.__outtopology['type']  = kwargs.get('OutTopologyType','binomial')
        
        self.__weights              = dict()
        self.__weights['distr']     = kwargs.get('ConnectionDistr','pm1')
        
        self.__adjecencylist        = list()
        self.__nodes                = self.InitializeNodes()
        self.__adjecency            = self.GenerateTopology()
        self.__connections          = self.__adjecency * self.GenerateConnectionStrength()
        
        self.__lastupdate_nodes     = -np.ones(self.__size,dtype=np.int)
        self.__lastupdate_input     = -np.ones(self.__size,dtype=np.int)
        self.__sInputBefore         = np.array(self.__size,dtype=np.float)
        
        self.__maxhistolength       = kwargs.get('MaxHistoLength')
        self.__updatehisto_nodes    = np.array([],dtype=np.int)
        self.__updatehisto_input    = np.array([],dtype=np.int)
        self.__condprobInput_total  = np.array([],dtype=np.float)
        self.__condprobInput_flip   = np.array([],dtype=np.float)
        self.__condprobNodes_total  = np.array([],dtype=np.float)
        self.__condprobNodes_flip   = np.array([],dtype=np.float)
        
        self.__histoXFafterSF       = np.array([],dtype=np.int)
        self.__histoSFafterXF       = np.array([],dtype=np.int)
        
        self.__countinputflips      = np.array([],dtype=np.int)
        
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

        # conditioned on how many steps passed since last flip
        self.UpdateCondProbNodesFlip(updateNodesHisto,update)
        self.UpdateCondProbInputFlip(updateInputHisto)
        
        # count how many flipped nodes occur in one input
        self.CountInputFlips(updateNodesHisto)
        
        # conditioned on a flip (in input or nodes), at what (earlier) times did either nodes (connecting to input) or input (of the current node) flip
        self.HistoNodesChangeAfterInputFlip(updateNodesHisto,updateInputHisto)
        self.HistoInputChangeAfterNodesFlip(updateNodesHisto,updateInputHisto)
        
        # set timestep of last update for changed inputs and nodes to current step
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
                self.__adjecencylist.append(connections)
                tmpadj[i][connections] = 1
        elif self.__intopology['type'] == 'full':
            tmpadj = np.ones((self.__size,self.__size),dtype = np.int)
            self.__adjecencylist = np.ones((self.__size,self.__size), dtype = np.int)
        else:
            raise NotImplementedError
        
        return tmpadj
    
    
    def GenerateConnectionStrength(self):
        tmpcs = np.zeros((self.__size,self.__size),dtype=np.float)
        if self.__weights['distr'] == 'pm1':
            tmpcs = np.random.choice([-1,1],size = (self.__size,self.__size))
        else:
            raise NotImplementedError
        return tmpcs
    
    
    def InitializeNodes(self):
        return 2 * np.random.binomial(self.__intopology['K'],.5,self.__size) - self.__intopology['K']
    
    
    def AddMaxHistoLength(self,steps):
        if self.__maxhistolength is None:
            # no maximal length defined, thus extend histograms indefintely
            return True
        else:
            if steps <= self.__maxhistolength:
                # max length not yet reached
                return True
            else:
                # max length reached, do not add additional bins to histograms
                return False
    
    
    def UpdateHistogram(self,histo,step):
        """
        general method to compute histograms
        includes 2 checks:
            * max length should not be exceeded
            * if the current length of the histogram is extended, more bins are needed
        """
        update = False
        if self.__maxhistolength is None:
            update = True
        else:
            if step <= self.__maxhistolength:
                update is True
        if update:
            if len(histo) <= step:
                histo = np.concatenate([histo,np.zeros(int(step - len(histo) + 1))])
            histo[step] += 1
        return histo
    
    
    def UpdateXHisto(self,updateNodesHisto,update):
        """
        compute P[xf,n]
        """
        countupdates = 0
        for nodeID in np.arange(self.__size)[updateNodesHisto]:
            StepsSinceUpdate = self.__step - self.__lastupdate_nodes[nodeID]
            
            if self.__lastupdate_nodes[nodeID] >= 0 and update[nodeID]:
                self.__updatehisto_nodes = self.UpdateHistogram(self.__updatehisto_nodes, StepsSinceUpdate)
                countupdates += 1
        
        return countupdates
    
    
    def UpdateSHisto(self,updateInputHisto):
        """
        compute P[sf,n]
        """
        countupdates = 0
        for nodeID in np.arange(self.__size)[updateInputHisto]:
            StepsSinceUpdate = self.__step - self.__lastupdate_input[nodeID]
            if self.__lastupdate_input[nodeID] >= 0:
                self.__updatehisto_input = self.UpdateHistogram(self.__updatehisto_input, StepsSinceUpdate)
                countupdates += 1
        return countupdates
    
    
    def UpdateCondProbInputFlip(self,updateInputHisto):
        """
        compute P[sf|n] by counting sf and n
        """
        for nodeID in range(self.__size):
            StepsSinceUpdate = self.__step - self.__lastupdate_input[nodeID]
            if self.__lastupdate_input[nodeID] >= 0:
                self.__condprobInput_total = self.UpdateHistogram(self.__condprobInput_total,StepsSinceUpdate)
                if updateInputHisto[nodeID]:
                    self.__condprobInput_flip = self.UpdateHistogram(self.__condprobInput_flip, StepsSinceUpdate)

    
    def UpdateCondProbNodesFlip(self,updateNodesHisto,update):
        """
        compute P[xf|n] by counting xf and n
        """
        for nodeID in range(self.__size):
            StepsSinceUpdate = self.__step - self.__lastupdate_nodes[nodeID]
            if self.__lastupdate_nodes[nodeID] >= 0:
                self.__condprobNodes_total = self.UpdateHistogram(self.__condprobNodes_total, StepsSinceUpdate)
                if updateNodesHisto[nodeID] and update[nodeID]:
                    self.__condprobNodes_flip = self.UpdateHistogram(self.__condprobNodes_flip, StepsSinceUpdate)

    
    def CountInputFlips(self,updateNodesHisto):
        """
        compute how many nodes of a single input are flipped this step
        """
        nodechanges  = np.where(updateNodesHisto, 1, 0)
        inputchanges = np.dot(self.__adjecency, nodechanges)
        for nodeID in range(self.__size):
            self.__countinputflips = self.UpdateHistogram(self.__countinputflips, inputchanges[nodeID])
        return np.sum(inputchanges)
    
    
    def HistoNodesChangeAfterInputFlip(self, updateNodesHisto, updateInputHisto):
        """
        compute P[xf,n|sf]
        """
        for nodeID in np.arange(self.__size)[updateNodesHisto]:
            StepsSinceUpdate = self.__step - self.__lastupdate_input[nodeID]
            if updateInputHisto[nodeID]: StepsSinceUpdate = 0
            self.__histoXFafterSF = self.UpdateHistogram(self.__histoXFafterSF, StepsSinceUpdate)
    
    
    def HistoInputChangeAfterNodesFlip(self, updateNodesHisto, updateInputHisto):
        """
        compute P[sf,n|xf]
        """
        for nodeID in np.arange(self.__size)[updateInputHisto]:
            for connected_nodeID in self.__adjecencylist[nodeID]:
                StepsSinceUpdate = self.__step - self.__lastupdate_nodes[connected_nodeID]
                if updateNodesHisto[connected_nodeID]: StepsSinceUpdate = 0
                self.__histoSFafterXF = self.UpdateHistogram(self.__histoSFafterXF, StepsSinceUpdate)
            
    
    
    def __getattr__(self,key):
        if   key == 'histoX':
            return self.__updatehisto_nodes
        elif key == 'histoS':
            return self.__updatehisto_input
        elif key == 'nodes':
            return self.__nodes
        elif key == 'connections':
            return self.__connections
        elif key == 'adjecency':
            return self.__adjecency
        elif key == 'condprobInput':
            return self.__condprobInput_flip,self.__condprobInput_total
        elif key == 'condprobNodes':
            return self.__condprobNodes_flip,self.__condprobNodes_total
        elif key == 'histoinputchange':
            return self.__countinputflips
        elif key == 'histoatflipnodes':
            return self.__histoXFafterSF
        elif key == 'histoatflipinput':
            return self.__histoSFafterXF


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
