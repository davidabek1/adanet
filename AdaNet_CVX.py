""" Created by David Abekasis March 2018
AdaNet: Adaptive Structural Learning of Artificial Neural Networks 
This API implements using algorithm based on the article: http://proceedings.mlr.press/v70/cortes17a.html

In short this model is building from scratch a neural network according to the data complexity it fits,
this is why it named as adaptive model. The problem at hand is always a binary classification.
During fit operation it will build the hidden layers and number of neurons in each layer. 
The decision if to go deeper (add hidden layer) or to go wider (add neuron to existing layer), 
or update an existing neuron weight is done in a closed form of calculations 
(by using Banach space duality) shown in the article. 
Lastly it will optimize the weight of the best neuron (added or existing), update parameters and iterate.

The article talks about several variants of AdaNet, this is the AdaNet.CVX implementation, 
explained on Appendix C - that solves a convex sub problem in each step in a closed form.

Further detailed explanations of this variant is shown in a previous version of the article [v.1]: 
all versions: https://arxiv.org/abs/1607.01097
v.1: https://arxiv.org/abs/1607.01097v1

I have used as a reference a MATLAB implementation of the algorithm from: https://github.com/lw394/adanet
Giving back to the community i've shared this project to: https://github.com/davidabek1/adanet

"""
import numpy as np
import copy

import scipy.optimize as op
import inspect

from sklearn.metrics import accuracy_score

class AdaNetCVX(object):
    def __init__(self,maxLayers=5,maxNodes=50,capitalLambda=10,pnorm=2.0,Ck=10.0,Ck_bias=10.0,\
                lowerLambda=0.001,beta=0.1,bolAugment=True,bolAugmentLayers=True,activationFunction='tanh',\
                T=50,lossFunction='binary',surrogateLoss='logistic',\
                numStableEpoch=30, minAvgLossChange=1e-8, optMethod='Nelder-Mead', optIsGrad=None):
        """ init of AdanNetCVX class as the classifier instance
        Args:
            maxLayers: int, default=5, max hidden layers that AdaNet can extend to
            maxNodes: int, default=50, max neurons that AdaNet can use on each hidden layer
            capitalLambda: float, default=10, hyper parameter of the lp norms of the weights defining new nodes
                           each layer has the same value, will be used to define cfg['maxWeightNorms'] param.
            pnorm: float, default=2, lp norm used to calculate the weights of new nodes
            Ck: float, default=10, upper bounds on the nodes in each layer
            Ck_bias: int, default=10, upper bounds intercept bias on the nodes in each layer
            lowerLambda: float, default=0.001, part of calculation of complexity penalties (regularizer) capital Gamma
                            used in reg method as Gamma = lambda*rj + beta.
                            The regularization term is a weighted-l1 penalty (Gamma*abs(w))
            beta: float, default=0.1, part of calculation of complexity penalties (regularizer) capital Gamma 
                            used in reg method as Gamma = lambda*rj + beta
                            The regularization term is a weighted-l1 penalty (Gamma*abs(w))
            bolAugment: boolean, default=True
            bolAugmentLayers: boolean, default=True
            activationFunction: str, default='tanh' (hyperbolic tan), activation function used at each node 
                                      for the hypothesis value
            T:          int, default=50, number of ephocs the model will run for convergence
            lossFunction: str, default='binary', the model is used as a binary classification problem, 
                              but this foundation is to be ready to implement future option of regression problem
                              will be used as 'MSE' value to calcualte mean squared error.
            surrogateLoss: str, default='logistic', the surrogate loss function is defined in the article 
                           as the capital phi, that is activated on the difference of the zero/one loss problem (1-y*(Sig(w*h)))
                           this is in order to be sure the sub problem is convex for optimization.
                           activation function is the logistic as exp(x)/(1+exp(x)).
                           alternative is the 'exp' function as , exp(x)
            numStableEpoch: int, default=30, over this number of epochs will be calculated the average of loss change, 
                            if it will show to be less than a threshold, a convergence will be assumed 
                            and the fitting iterations will stop.
            minAvgLossChange: float, default=1e-8, this is the threshold of the average loss change, 
                             over a stable number of epochs. if the average is not higher than this value, 
                             convergence will be assumed and the fitting iterations will stop.
            optMethod: str, default='Nelder-Mead' the method that the optimizer will use, 
                            for the step search of best node weight
            optIsGrad: boolean, default=None, if to ask for the gradient from the loss function, 
                            if required by some optimizers methods, if yes a True value is required, 
                            None value otherwise.

        """
        self.name = 'AdaNet.CVX model'
        self.maxLayers = maxLayers
        self.maxNodes=maxNodes
        self.capitalLambda=capitalLambda
        self.pnorm=pnorm
        self.Ck=Ck
        self.Ck_bias=Ck_bias
        self.lowerLambda=lowerLambda
        self.beta=beta
        self.bolAugment=bolAugment
        self.bolAugmentLayers=bolAugmentLayers
        self.activationFunction=activationFunction
        self.T=T
        self.lossFunction=lossFunction
        self.surrogateLoss=surrogateLoss
        self.numStableEpoch=numStableEpoch
        self.minAvgLossChange=minAvgLossChange
        self.optMethod=optMethod
        self.optIsGrad=optIsGrad
        self._init_cfg()

    def _init_cfg(self):
        """inner helper to initialize the configuration dictionary (cfg)
        based upon the model is working
        """
        self.cfg = {}
        self.cfg['maxNodes'] = np.dot(np.array(self.maxNodes),np.ones((1,self.maxLayers)))
        self.cfg['maxWeightNorms'] = (self.capitalLambda*np.ones(self.cfg['maxNodes'].shape)).reshape(-1,1) # capital lambda
        self.cfg['pnorm'] = self.pnorm
        self.cfg['maxWeightMag'] = (self.Ck*np.ones(self.cfg['maxNodes'].shape)).reshape(-1,1) # Ck
        self.cfg['maxBiasMag'] = self.Ck_bias # Ck bias
        self.cfg['complexityRegWeight'] = self.lowerLambda*np.ones(max(self.cfg['maxNodes'].shape)) # lower Lambda
        self.cfg['normRegWeight'] = (self.beta*np.ones(self.cfg['maxNodes'].shape)).reshape(-1,1) # beta
        self.cfg['augment'] = self.bolAugment
        self.cfg['augmentLayers'] = self.bolAugmentLayers
        self.cfg['activationFunction'] = self.activationFunction 
        self.cfg['numEpochs'] = self.T
        self.cfg['lossFunction'] = self.lossFunction
        self.cfg['surrogateLoss'] = self.surrogateLoss

    def __str__(self):
        """Implementing the str method to return classifier name
        """
        return "Classifier: " + self.name

    def get_params(self, cfgout=False):
        """Get parameters for this estimator.
        Args:
            cfgout (boolean): if to return AdaNet init configuration based on input params
                           as another dictionary default to False
        Returns:
            params (dict): mapping of current input parameter names mapped to their values.
                           if set_params() not used, will return the parameters the classifier was initialized
            cfg (dict): based on input parameters a configuration dictionary is created 
                        based upon the model is working
        """
        init_signature = inspect.signature(self.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        params_names = sorted([p.name for p in parameters])
        params = dict()
        for key in params_names:
            value = getattr(self, key, None)
            params[key]=value
        if cfgout:
            return params, self.cfg
        else:
            return params

    def set_params(self, **params):
        """Set the parameters of this estimator.
        Args:
            **params (kwargs): a dictionary like list of parameters and their values, 
                            e.g. T=1000 will set the number of epochs the model will run
        Returns:
                self
        """
        if not params:
            return self

        valid_params = self.get_params(cfgout=False)

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter {} for estimator {}. '
                                    'Check the list of available parameters '
                                    'with `estimator.get_params().keys()`.'.format(key, self))
            else:
                setattr(self, key, value)
        return self


    def _adanet(self, Xdata,ydata, cfg):
        '''Main method for AdaNet.CVX
        This method encapsulating all parts needed to learning the structure of the adapted neural network.
        It is using inner methods, in order to simplify input and output arguments, 
        while those inner methods are used only during the fit operation.
        Inner methods include: 
            - ExistingNodes() to calc the value of updating weights of existing nodes
            - NewNode() to calc the value of adding a new node to each of existing hidden layers 
                or a starting a new one.
            - BestNode() - to find from the above methods what should be next step in building the network,
                either fine tune existing weights, or adding a neuron (node) 
                either on one of existing layers or open a new hidden layer
            - ApplyStep() - for the best node selected in BestNode() calc the weight based on line search optimization
            - UpdateDistribution() - this method updates the distribution over the sample as input for the next iteration
        Args:
            Xdata: Ndarray [m_samples, n_features]
                    input train data
            ydata: Ndarray [m_samples,1]
                    target train labels in [-1,1]
            cfg:   dict, base configuration hyper parameters of the model 
        Returns:
            adaParams: dict, learned architecture and parameters of the network,
                          will be used during predict and score functions.
            history: dict, saved snapshots of learned network during fit operation, 
                           for use of debuging    
        '''
        def ExistingNodes():
            ''' Existing-nodes function
            This method calculate the value of updating weights of existing nodes d_kj (also intercept value d_bias), 
            this value will be compared to new nodes method values (in the next method NewNodes()), 
            and decide best approach minimizing loss function in the BestNode() method.
            Returns:
            d_kj: dict, each key represent existing hidden layer {0...numLayers-1}
                        and the dict value is a NDarray [numNodes,1] representing for each node the score value,
                        which is higher is better for comapring alternatives of using new nodes or existing ones.
            d_bias: NDarray [1,1], the intercept value (bias) for the d_kj above
            '''
            errLoss = adaParams['errLoss']
            Dloss = adaParams['D'][lossIdcs]
            Dloss = Dloss/Dloss.sum(axis=0) 
            m = max(lossIdcs.shape) 
            d_kj = {}     
            # initial scores for all nodes
            for k in range(numLayers): 
                d_kj[k] = np.zeros((numNodes[k,0],1))  # support for line 7
                
            d_bias = np.array([[0.]]) 
            
            S = adaParams['S']
            for k in range(numLayers):  
                Ck = cfg['maxWeightMag'][k] 
                
                if cfg['lossFunction'] == 'binary':
                    # errLoss replacing only relevant updated values leaving the rest with zeros
                    existingNodes2Update = H[k].shape[1]
                    errLoss[k][0:existingNodes2Update,:] = Ck/2 * (1-1/Ck*np.dot(((ydata[lossIdcs].reshape(-1,1)*self._activation(H[k][lossIdcs,:],actFunc)).conj().T),Dloss)) 
                    
                # %                 case 'MSE'
                # %                     errLoss{k} = Ck/2 *(1 - 1/Ck * (abs(bsxfun(@minus, ydata(lossIdcs,:),H{k}(lossIdcs,:))).^2)'*Dloss);
                else:
                    raise Exception('loss error','loss not supported')
                
                Wk = W[k][:numNodes[k,0]] 
                eLk = errLoss[k][:numNodes[k,0]]
                # nodes with non-zero weights
                j_nzw = Wk != 0  #pg 7, Existing Nodes, line 4
                # nodes with scores of zero
                j_zs = np.logical_and(np.logical_not(j_nzw) , (np.abs(eLk - Ck/2) <= Rloss[k]*m/(2*S))) # pg 7, Existing Nodes, line 6
                # remaining nodes
                j_other = np.logical_and(np.logical_not(j_nzw),np.logical_not(j_zs)) # line 8
                # compute and assign existing node scores
                d_kj[k][j_other] = eLk[j_other] - Ck/2 - np.sign(eLk[j_other]-Ck/2)*Rloss[k]*m/(2*S)
                d_kj[k][j_nzw] = eLk[j_nzw] - Ck/2 + np.sign(Wk[j_nzw])*Rloss[k]*m/(2*S)

            Ck_bias = adaParams['maxBiasMag']
            # inference using existing parameters
            
            if cfg['lossFunction'] == 'binary':
                errLoss_bias = Ck_bias/2 * (1-1/Ck_bias*np.dot(((ydata[lossIdcs].reshape(-1,1)*np.ones((ydata[lossIdcs].shape[0],1))).conj().T),Dloss))
            # %             case 'MSE'
            # %                 errLoss_bias = Ck_bias/2 * (1-1/Ck_bias*(abs(bsxfun(@minus, ydata(lossIdcs,:),ones(size(ydata(lossIdcs,:))))).^2')*Dloss);
            else:
                raise Exception('loss error','loss not supported')
            
            Wk_bias = adaParams['W_bias']
            # nodes with non-zero weights
            j_nzw = Wk_bias != 0  # pg 7, Existing Nodes, line 4
            # nodes with scores of zero
            j_zs = np.logical_and(np.logical_not(j_nzw) , (np.abs(errLoss_bias - Ck_bias/2) <= 0))  # pg 7, Existing Nodes, line 6
            # remaining nodes
            j_other = np.logical_and(np.logical_not(j_nzw),np.logical_not(j_zs))   # line 8
            # compute and assign existing node scores
            d_bias[j_other] = errLoss_bias[j_other] - Ck_bias/2 - 0 
            d_bias[j_nzw] = errLoss_bias[j_nzw] - Ck_bias/2 + 0 
            return d_kj,d_bias

        def NewNodes():
            ''' New nodes function
            This method calculate the score value (dn_k) for adding new nodes on each of the existing layers
            (while max nodes on the layer has not yet reached), 
            and also what is the score value of adding a node on a possible new hidden layer 
            (again while max hidden layer has not been reached).
            This values including the previous calculations of existing nodes, 
            will be compared in the next function of BestNode()
            Returns:
            dn_k: dict, each key represent existing hidden layer + 1 optional new hidden layer {0...numLayers}
                        and the dict value is a NDarray [1,1] representing for each new node on relevant hidden layer
                        the score value, which is higher is better for comapring alternatives of using new nodes or existing ones.
            un_k: dict, each key represent existing hidden layer + 1 optional new hidden layer {0...numLayers}
                        and the dict value is a NDarray [numNodes(k-1),1] representing the weights coming from previous layer (k-1)          
            '''
            global errNew
            numNodes = adaParams['numNodes']
            maxNodes = adaParams['maxNodes']  # shape[1,5] 
            q = adaParams['qnorm']
            p = cfg['pnorm']
            idx = np.minimum(numLayers+1,adaParams['maxLayers'])
            errNew = {} 
            un_k = {} 
            dn_k = {} 

            Dnew = adaParams['D'][newIdcs,0] 
            Dnew = Dnew/Dnew.sum(axis=0) 
            m = max(newIdcs.shape) 
            S = adaParams['S']  # number of samples

            for k in range (idx):   # pg 8, Fig 4, Line 1
                Ck = cfg['maxWeightMag'][k]  # shape[5,]
                Lambda_k = cfg['maxWeightNorms'][k]   # shape[5,]

                if k==0:   
                    nodeNum_km1 = numInputNodes  
                    Hk = Xdata  
                    actFunc = 'none'
                else:
                    nodeNum_km1 = numNodes[k-1,0]  
                    Hk = H[k-1]  
                    actFunc = adaParams['activationFunction']

                if (numNodes[k,0] < maxNodes[0,k]) and (nodeNum_km1 > 0):  # pg 8, Fig 4, line 2
                    if adaParams['augmentLayers']:
                        Hk = np.hstack((np.ones((Hk.shape[0],1)),Hk))  
        
                    if cfg['lossFunction'] == 'binary': 
                        # M short for Margin
                        # M_kminus1 is the vector of weighted margin of all nodes on layer k-1 as 
                        #           hypothesis H[k-1] composed with its activation on distribution D 
                        #           Ndarray[m newIdcs samples,1]
                        M_kminus1 = (np.dot(((ydata[newIdcs].reshape(-1,1)*self._activation(Hk[newIdcs],actFunc)).conj().T),Dnew)).reshape(-1,1) 
                        M_qnorm = np.linalg.norm(M_kminus1,q)   
                        # errNew[k] is the weighted error of adding a node to layer k 
                        #           based on the weighted margin of k-1 layer  
                        #           used as the basis to calc the score of adding a node to layer k                 
                        errNew[k] = Ck/2*(1-Lambda_k/Ck * M_qnorm)  # pg 8, Fig 4, line 3
        #%                     case 'MSE'
        #%                         M_kminus1 =  (abs(bsxfun(@minus,ydata(newIdcs,:),activation(Hk(newIdcs,:),actFunc))).^2)'*Dnew;
        #%                         M_qnorm = norm(M_kminus1,q);
        #%                         errNew{k} = Ck/2*(1-Lambda_k/Ck * M_qnorm);  % pg 8, Fig 4, line 3


                    if p == 1:
                        _,bMax = (np.abs(M_kminus1)).max(),(np.abs(M_kminus1)).argmax() 
                        M_normalized = np.zeros(M_kminus1.shape)  
                        M_normalized[bMax] = 1/np.abs(M_kminus1[bMax])  
                        un_k[k] = Lambda_k * M_normalized*np.sign(M_kminus1)  
                    else:
                        un_k[k] =  Lambda_k * (np.abs(M_kminus1)**(q-1))*np.sign(M_kminus1)/(M_qnorm**(q/p)) # line 4

                else:
                    errNew[k] = 1/2*Ck   # line 5,


                if np.abs(errNew[k] - 1/2*Ck) <= Rnew[k]*m/(2*S):  # line 6
                    dn_k[k] = 0  # line 7
                else:
                    dn_k[k] = errNew[k] - Ck/2 - np.sign(errNew[k] - Ck/2)*Rnew[k]*m/(2*S) # line 8
            return dn_k, un_k

        def BestNode():
            ''' Best node function
            This method examine the values coming from ExistingNodes() and NewNodes(), 
            and select the best approach for this iteration, based on this decision, 
            if decide to add a new node to the network, it will calculate relevant information of the new node
            as updated loss value, what is the state of network regarding of number of hidden layers, 
            and number of nodes on each hidden layer. 
            if decide to stay with existing nodes it will only point to best node in a layer as jk_best
            Returns:
            jk_best: str ('bias') if selected best approach is only update bias, 
                    or Ndarray [1,1] => [location of best node, on which layer this node resides]
            e_t:  dict, each key represent existing hidden layer + 1 optional new hidden layer {0...numLayers}
                    the value at each key is Ndarray of [Number of Nodes on this layer,1],
                    while each value represent the error loss of this node
            numLayers: int, current number of layers in the structured network, 
                       will be updated, if by best approach the new node will be selected on a new hidden layer
            numNodes: Ndarray [maxNumLayers,1], current number of nodes on each of hidden layers, 
                       will be updated, if by best approach a new will be selected, either on existing or new layer,
                       number of nodes will be updated.
                       the gap from existing number of layers to max allowed layers, is carying 0 as number of nodes.
            '''       
            global errNew
            errLoss = adaParams['errLoss']
            numLayers = adaParams['numLayers']
            numNodes = adaParams['numNodes']
            # calculate score for output bias
            d_biasMax = np.abs(d_bias) 
            # get most important new node that could exist if we augment the network
            dnAbs = {}                 
            for k,v in dn_k.items():   
                dnAbs[k] = np.abs(v)   
            
            dnMaxk = np.zeros((len(dnAbs.keys()),1))            
            
            for k,v in dnAbs.items():                           
                dnMaxk[k] = v.max()   
            dnMax = dnMaxk.max()      
            kBinNew = dnMaxk.argmax()
            # get most important node in all current layers
            if not(d_kj):  # should only happen on the first 
                dMax = dnMax-1  # ust make sure dMax is less than dnMax
            else:
                dAbs = {}                  
                for k,v in d_kj.items():   
                    dAbs[k] = np.abs(v) 
                
                dMaxk = np.zeros((len(dAbs.keys()),1))
                for k,v in dAbs.items():
                    dMaxk[k] = v.max()    
                kBin = dMaxk.argmax()                               
                dMax, jBin = dAbs[kBin].max(), dAbs[kBin].argmax() 

            dcase = np.array([d_biasMax, dMax, dnMax]).argmax()
            
            # Pick the highest score and possibly create a new node/layer
            if dcase == 0:
                jk_best = 'bias'
            elif dcase == 1:
                jk_best = np.array([jBin,kBin]) # line 2
            else:
                kNew = kBinNew                  # line 3
                if kNew > numLayers-1:          # line 4
                    numLayers = numLayers + 1   # line 5
                    history['newL'] = t

                numNodes[kNew] = numNodes[kNew] + 1  # line 6
                # converting j_best into zero index by minus 1
                jk_best = np.array([numNodes[kNew,0]-1,kNew])  # line 7
                history['newN'].append(np.hstack((np.array([t]).reshape(1,-1),np.zeros((1,adaParams['maxLayers']))))) 
                history['newN'][-1][0,kNew+1] = 1  
                history['jk_best'].append(jk_best) 
                j_best = jk_best[0] 
                # Augment u appropriately
                # adding a new column resembling the u values comming from all existing nodes of lower layer 
                # into this new node
                if not(u) or not(kNew in u):
                    # cannot append to empty matrix, so initialize with first created un_k
                    u[kNew] = un_k[kNew]
                else:
                    u[kNew] = np.append(u[kNew],un_k[kNew],axis=1)
                if kNew+1 in u:
                    # because of adding a node in a middle/lower hidden layer, 
                    # the next layer has to be added with a corresponding row of zeros, 
                    # resembling that there is no connection between this new node to the upper layer
                    u[kNew+1] = np.append(u[kNew+1], np.zeros((1,u[kNew+1].shape[1])),axis=0) 

                # Update H with new added node
                if kNew == 0:  #if first hidden layer, Hk representing features set as Xdata
                    Hk = Xdata
                    actFunc = 'none'
                else:  #if second and above hidden layer, Hk is H[kNew-1]
                    Hk = H[kNew-1]  
                    actFunc = adaParams['activationFunction']

                if adaParams['augmentLayers']:
                    Hk = np.hstack((np.ones((Hk.shape[0],1)),Hk)) 

                # adding a new column resembling the new node values of h, 
                # while rows are the number of samples data
                if not(H) or not(kNew in H): # cannot append column into non existent array
                    H[kNew] = (np.dot(self._activation(Hk,actFunc),un_k[kNew])).reshape(-1,1)
                else:
                    H[kNew] = np.append(H[kNew], np.dot(self._activation(Hk,actFunc),un_k[kNew]), axis=1) 
                
                # Update error loss
                errLoss[kNew][j_best,0] = errNew[kNew]   

            e_t = errLoss   #  line 9
            return jk_best, e_t, numLayers, numNodes


        def ApplyStep(jk_best, surrogateLoss):
            ''' Apply step
            This method will for the best node selected in BestNode() calculate 
            the new weight of the selected node based on line search optimization
            Args:
            jk_best: str ('bias') if selected best approach is only update bias, 
                    or Ndarray [1,1] => [location of best node, on which layer this node resides]
            surrogateLoss: str, 'logistic' or 'exp' used as the capital phi(-x) in the article to impose 
                         a non-increasing convex function upper bounding the 0/1 loss w.r.t W (weight).
                         can be exponential phi(x)=exp(x), or logistic function phi(x)=log(1+exp(x))
            Returns:
            Wt: dict, each key represent existing hidden layer {0...numLayers-1}
                        and the dict value is a NDarray [numNodes,1] representing for each node the weight value,
                        considered as the weight of the node hypothesis.

            TODO:
            Optimizer is using Nelder-Mead solver using an approximate gradient numerically 
                      without the use of a gradient (jacobian), 
                      while testing with BFGS solver using the gradient in the optimizer, 
                      after certain amount of epochs the solver throws failure to optimize, 
                      looking for solution suggest maybe the values of the gradient goes negative, 
                      i couldn't figure out to resolve, so went for the Nelder-Mead solver without a gradient.

            Same phenomenon is happening with Nelder-Mead, although less frequent, 
                    my workaround is to reverse to last known good data, stop iterations of convergence,
                    and return with that results.

            Comment on MATLAB reference code to Normalize sum of weights to 1?? 
                As this is part of a proof in the paper (which I couldn't find), 
                As the refernce was corresponding with one of the paper's authors, 
                it could come from that.
            '''
            if str(jk_best) == 'bias':    
                w_k0 = adaParams['W_bias']
                h_k = np.array([[1.]])
                reg_k = 0
                
                f_notk = 0
                reg_notk = 0
                for k in range(numLayers):    
                    for j in range(numNodes[k,0]): 
                        f_notk = f_notk + np.dot(H[k][:,j].reshape(-1,1),W[k][j].reshape(-1,1))  
                        reg_notk = reg_notk + np.dot(Rloss[k],np.abs(W[k][j])) 

                if cfg['lossFunction'] == 'binary':
                    loss_notk = ydata*f_notk 
                # %                 case 'MSE'
                # %                     loss_notk = abs(ydata-f_notk).^2;

            else:
                # apply step
                # jk_best is the coordinate that we're going to tweak, and
                # we need to determine the step size
                j_best = jk_best[0] 
                k_best = jk_best[1]
                w_k0 = (W[k_best][j_best,0]).reshape(-1,1) 
               
                h_k = H[k_best][:,j_best].reshape(-1,1) 
                reg_k = Rloss[k_best] 
                
                f_notk = 0
                reg_notk = 0
                for k in range(numLayers):   
                    for j in range(numNodes[k,0]): 
                        if (k==k_best) and (j == j_best): 
                            pass
                        else:
                            f_notk = f_notk + np.dot(H[k][:,j].reshape(-1,1),W[k][j].reshape(-1,1)) 
                            reg_notk = reg_notk + np.dot(Rloss[k],np.abs(W[k][j])) 

                if cfg['lossFunction'] == 'binary':
                    loss_notk = ydata*(f_notk+adaParams['W_bias']) 
                # %                 case 'MSE'
                # %                     loss_notk = abs(ydata-f_notk - adaParams.W_bias).^2;
            
            # set up optimizer
            Result = op.minimize(fun = self._loss_function, 
                                            x0 = w_k0, 
                                            args = (h_k.reshape(-1,1), reg_k, ydata, loss_notk,reg_notk, surrogateLoss,cfg['lossFunction']),
                                            method = self.optMethod,  #'Nelder-Mead', 'BFGS'
                                            jac = self.optIsGrad  #None ,True
                                            )
            w_k = Result.x
            # check that optimizer succeeded
            if not(Result.success):
                print('Optimizer Failure: Nelder-Mead optimizer did not succeed')
                raise Exception('Optimizer Failure', 'Nelder-Mead optimizer did not succeed')
            
            if str(jk_best) == 'bias':    
                adaParams['W_bias'] = w_k
                history['W_bias'].append(w_k)
                Wt = copy.deepcopy(W) 
            else:
                # Overwrite weight at the (k,j)th node being considered
                W[k_best][j_best,0] = w_k  

                Wt = copy.deepcopy(W) 
                history['Wt'].append(Wt)
            return Wt


        def updateDistribution(Wt):
            ''' Update Distributions
            This method will update the maintained distibution on ephoc t,
            over the sample (Xdata is a sample of the distribution D).
            The distribution is a gradient of the objective loss function,
            and normalized over the sample (overhaul sum of sample grads).
            Args:
            Wt: dict, each key represent existing hidden layer {0...numLayers-1}
                        and the dict value is a NDarray [numNodes,1] representing for each node the weight value,
                        considered as the weight of the node hypothesis.
            Returns:
            Dnew: Ndarray [m_samples,1], the maintained distribution over the sampled data for the current ephoc t,
                  our dataset is a sample from the distribution D, that we try to estimate.
                  calculated as the gradient of the objective loss function.
            Snew: Ndarray [1,1], the normalization factor of the distribution, 
                  as the overhaul sum of each distribution value of the sampled dataset
            '''
            fNew = 0
            for k in range(numLayers):    
                for j in range(numNodes[k,0]):   
                    fNew = fNew + np.dot(H[k][:,j].reshape(-1,1),Wt[k][j].reshape(-1,1))  

            if cfg['lossFunction'] == 'binary':
                gradArg = 1 - ydata*(fNew+adaParams['W_bias']) 
            # %             case 'MSE'
            # %                 gradArg = abs(ydata-(fNew+adaParams.W_bias)).^2;

            phiGrad = self._slgrad(gradArg,adaParams['surrogateLoss']) 
            # update the sum and the distribution
            Snew = phiGrad.sum(axis=0) 
            Dnew = phiGrad/Snew
            return Dnew, Snew
        

        history = {}
        history['newL'] = []
        history['numL'] = []
        history['newN'] = []
        history['numN'] = []
        history['jk_best'] = []
        history['Wt'] = []
        history['ut'] = []
        history['W_bias'] = []
        history['activationFunction'] = cfg['activationFunction']

        if cfg['augment']:
            Xdata = np.hstack((np.ones((Xdata.shape[0],1)),Xdata))      
        if cfg['augment'] and cfg['augmentLayers']:
            Xdata = Xdata[:,1:] # revert Xdata! We will account for it in NewNodes

        numExamples = Xdata.shape[0] # number of training examples (m)
        numInputNodes = Xdata.shape[1] # number of features of the data

        if ydata.shape[0] != numExamples: 
            raise Exception('dim check', 'Xdata and ydata must have same number of examples (i.e. rows)')

        adaParams = self._adanet_init(numExamples,numInputNodes,cfg) 

        adaParams['lossStore'] = []
        adaParams['numInputNodes'] = numInputNodes
        adaParams['numOutputNodes'] = 1 #As a binary problem ydata.shape[1] has only 1dim, in cases of more labels will have relevant value
        T = cfg['numEpochs'] # num of iterations from paper

        xloss_end = round(numExamples/2)

        for t in range(T):
            idcs_t = np.random.permutation(range(numExamples)) # shuffle samples uniformly every epoch
        # split data into two mini batches, one for loss, one for finding new
        # nodes (Appendix E, page 17)
            lossIdcs = idcs_t[:xloss_end] 
            newIdcs = idcs_t[xloss_end:] 
        
        # Compute Forward Pass + Complexity Measures
            numLayers = adaParams['numLayers']
        
            H = {} # Hypothesis value per hidden layer
            Rloss = np.zeros((numLayers+1,1)) # complexity measures per layer
            Rnew = Rloss.copy()
            W = adaParams['W']
            u = adaParams['u']
            for k in range(numLayers):   
                if k == 0: 
                    Hk = Xdata
                    actFunc = 'none'
                else:
                    Hk = H[k-1]   
                    actFunc = adaParams['activationFunction']

                if adaParams['augmentLayers']:
                    Hk = np.hstack((np.ones((Hk.shape[0],1)),Hk))  
                    
                H[k] = np.dot(self._activation(Hk,actFunc),u[k])  
                # Calculate loss function
                if cfg['lossFunction'] == 'binary':
                    Rloss[k] = self.RademacherComplexity(H[k][lossIdcs,:])  
                    Rnew[k] = self.RademacherComplexity(H[k][newIdcs,:])  
    # %             case 'MSE'
    # %                 Rloss(k) = GaussianComplexity(H{k}(lossIdcs,:)); %equivalent with gaussian, instead of binary, noise
    # %                 Rnew(k) = GaussianComplexity(H{k}(newIdcs,:));
                else:
                    raise Exception('loss error', 'loss not supported')

                Rloss[k] = self.reg(Rloss[k],adaParams['complexityRegWeight'][k],adaParams['normRegWeight'][k]) #Rloss(k) = reg(Rloss(k),adaParams.complexityRegWeight(k),adaParams.normRegWeight(k));
                Rnew[k] = self.reg(Rnew[k],adaParams['complexityRegWeight'][k],adaParams['normRegWeight'][k])  #Rnew(k) = reg(Rnew(k),adaParams.complexityRegWeight(k),adaParams.normRegWeight(k));

            # Fallback values of adaParams and history
            # in case of exception during optimization phase
            fallback_adaParams = copy.deepcopy(adaParams)
            fallback_history = copy.deepcopy(history)
            # score the existing nodes in the network
            d_kj, d_bias = ExistingNodes()   #  pg 7, line 3
            dn_k, un_k = NewNodes()      # pg 7, line 4
            jk_best, e_t, numLayers, numNodes = BestNode()   # pg 7, line 5
            adaParams['numLayers'] = numLayers
            adaParams['numNodes'] = numNodes
            adaParams['u'] = u
            # Handle not finding an optimize step for the weight, 
            # so stop fiting the network, and use it at that point.
            try:
            # Move forward a step and update distributions
                Wt = ApplyStep(jk_best,adaParams['surrogateLoss'])
            except:
                adaParams = fallback_adaParams
                history = fallback_history
                return adaParams,history

            Dt,St = updateDistribution(Wt)
            # Pack things back into adaParams
            adaParams['D'] = Dt
            adaParams['S'] = St
            adaParams['W'] = Wt 
            adaParams['errLoss'] = e_t

            # Store history for debugging only
            history['Wt'].append(adaParams['W'])   
            history['ut'].append(adaParams['u'])  
            history['numL'].append(numLayers)    
            history['numN'].append(adaParams['numNodes'].copy()) 

            fComplete = 0
            for k in range(numLayers):   
                for j in range(numNodes[k,0]): 
                    fComplete = fComplete + np.dot(H[k][:,j].reshape(-1,1),W[k][j].reshape(-1,1)) 

            fComplete = fComplete + adaParams['W_bias']
            if cfg['lossFunction'] == 'binary':
                adaParams['lossStore'].append((self._slfunc(1-ydata*fComplete, adaParams['surrogateLoss'])).mean(axis=0))
        # %         case 'MSE'
        # %             adaParams.lossStore(end+1) =mean(abs(ydata - fComplete).^2);
        
        # Check for convergence based on change in last numStableEpoch itertions
            numStableEpochToAvg = min(self.numStableEpoch, len(adaParams['lossStore']))
            if t>1 and np.abs(np.mean(np.diff(np.hstack(adaParams['lossStore'][-numStableEpochToAvg:])))) <= self.minAvgLossChange:
                return adaParams,history        
        return adaParams,history


    def _activation(self,h,afunc):
        ''' Activation function
        The node activation function used in the network over the hypothesis value
        Args:
            h: Ndarray[m_samples, n_nodes] hypothesis value to activate upon the crushing function
            afunc: str, the activation function either 'relu', 'tanh' or 'sigmoid'
        Returns:
            a: Ndarray[m_samples, n_nodes] the h value after activation
        '''
        choices = {'relu': np.maximum(0,h), 'tanh': np.tanh(h), 'sigmoid': 1./(1+np.exp(-h))}
        a = choices.get(afunc, h)
        return a


    def RademacherComplexity(self,H):
        '''Rademacher Complexity function
        This method computes the Rademacher complexity over the hypothesis H, 
        Rademacher is a distribution with equally probability of 0.5 for -1 and +1 values.
        in order to add regularization to the objective loss function to become a coercive convex function.
        Args:
            H: Ndarray[m_samples, n_nodes] the hypothesis value to add Rademacher noise to it
        Returns:
            R: Ndarray[m_samples, n_nodes] the hypothesis value after the Rademacher noise added to it
        '''
        M = H.shape[0] 
        radNoise = np.random.standard_normal((M,1)) 
        radNoise[radNoise <= 0] = -1
        radNoise[radNoise > 0] = 1

        R = 1/M*np.dot(radNoise.conj().T,H.sum(axis=1))
        return R


    def GaussianComplexity(self,H):
        '''Gaussian Complexity function
        This function computes the Gaussian complexity over the hypothesis H, 
        The Gaussian distribution is the normal distribution with mean=0 and sigma=1
        in order to add regularization to the objective loss function to become a coercive convex function.
        Args:
            H: Ndarray[m_samples, n_nodes] the hypothesis value to add Gaussian noise to it
        Returns:
            R: Ndarray[m_samples, n_nodes] the hypothesis value after the Gaussian noise added to it
        '''
        M = H.shape[0] 
        noise = np.random.standard_normal((M,1)) 

        R = 1/M*np.dot(noise.conj().T,H.sum(axis=1))
        return R


    def reg(self,R,lam,beta):
        '''Regularization parameter
        This function calculates the Capital Gamma of the object loss function, as the regularization parameter
        Args:
            R: Ndarray[1, ] the hypothesis value after a noise distribution added to it
                the H value sent to this function is the overhaul value of H on the k layer
            lam: float, lower lambda as a hyper parameter for the regularization term,
                it will multiply r value (noised h) with the lambda parameter
        beta: Ndarray[1, ] as a hyper parameter for the regularization term,
                it will be added to the multiplied term to construct gamma value
        Returns:
            gamma: Ndarray[1, ] the capital gamma value as the multiplier of the regularization term with the weights
                weighted l1 penalty (abs of weights).
                Capital_GAMMA = lowerLambda*rj+beta
        '''
        gamma = np.dot(lam,R)+beta 
        return gamma


    def _adanet_init(self,numExamples,numInputNodes,cfg):
        ''' Initialization function
        implementation of Init method, Figure 5,  pg 15
        This method 
        Args:
            numExamples: int, number of samples (rows) provided by the input parameter X in fit() method
            numInputNodes: int, number of features (columns) provided by the input parameter X in fit() method
            cfg: dict, configuration dictionary based on input parameters,
                        or during set_parmas() method the configuration parameters can be set, 
                        based upon the model is working.
        Returns:
            AdaInit: dict, extended initialized configuration dictionary based on the specific data provided,
                        during the fit operation, the values in the dictionary will be updated, 
                        according to the model.
                        examples: numNodes Ndarray [maxNodes,1] is initialized to zero 
                                  on each possible hidden layer (maxNodes). 
                                  during fit, added nodes will update the number of nodes 
                                  on each of created layers by the model.
        '''
        # Initialize output weights (w)
        maxNodes = cfg['maxNodes']  # shape[1,maxNodes]
        maxLayers = max(cfg['maxNodes'].shape) 
        W = {} 
        for k in range(maxLayers): 
            W[k] = np.zeros((np.int(maxNodes[0,k]),1)) 

        # Assign current number of nodes and layers
        numLayers = 0    # number of hidden layers (no input or output layer considered)
        numNodes = np.zeros((max(maxNodes.shape),1),dtype=np.int) # shape[maxLayers,1] number of nodes in each hidden layers

        # Initialize feed-in weights (u)
        u = {} 
        # Create initial distribution (uniform)
        D = np.dot(1/numExamples, np.ones((numExamples,1))) 

        adaInit = {}

        adaInit['W'] = W
        adaInit['W_bias'] = np.array([[0.]])  # shape[1,1]
        adaInit['u'] = u
        adaInit['D'] = D
        adaInit['S'] = numExamples
        adaInit['errLoss'] = copy.deepcopy(W)
        adaInit['numLayers'] = numLayers
        adaInit['maxLayers']= maxLayers
        adaInit['numNodes'] = numNodes
        adaInit['maxNodes'] = maxNodes
        adaInit['qnorm'] = 1/(1-1/cfg['pnorm']) 
        adaInit['complexityRegWeight'] = cfg['complexityRegWeight']  # lambda
        adaInit['normRegWeight'] = cfg['normRegWeight']  # beta
        adaInit['surrogateLoss'] = cfg['surrogateLoss']
        adaInit['lossFunction'] = cfg['lossFunction']
        adaInit['maxWeightMag'] = cfg['maxWeightMag']  # Ck
        adaInit['maxBiasMag'] = cfg['maxBiasMag']  # Ck_bias
        adaInit['activationFunction'] = cfg['activationFunction']
        adaInit['augment'] = cfg['augment']
        adaInit['augmentLayers'] = cfg['augmentLayers']
        
        return adaInit


    def _loss_function(self,w_k, h_k, reg_k, y, loss_notk,reg_notk, surrloss,lossfunc):
        ''' Loss Function
        implementation of Coordinate descent described on pg 6
        This method calculate the objective loss function of the model, 
        supporting the coordinate descent operation (optimizer) for the line search optimization 
        to update the weight of the best node, 
        that will maximize convergence of the model (minimizing the loss function)
        Args:
            w_k: float, the current weight of the best node selected to find a step update (coordinate descent)
                 that will minimize the loss function
            h_k: Ndarray [m_samples,1], the hypothesis values of the best node by each of the samples.
            reg_k:  Ndarray[1,1], the Gamma value of the best node layer
            y: Ndarray[m_samples,1], the target values
            loss_notk: Ndarray[m_samples,1] the "loss" of all but the best node, 
                        implement yi*ft-1(xi) part of the loss function
            reg_notk: Ndarray[1,1], the sum of regularizer term of all but the best node layer,
                                    implemented as all but best node => Sigma(Gamma*W)

            surrloss: str, default='logistic', the surrogate loss function is defined in the article 
                           as the capital phi, that is activated on the difference of the zero/one loss problem (1-y*(Sig(w*h)))
                           this is in order to be sure the sub problem is convex for optimization.
                           activation function is the logistic as exp(x)/(1+exp(x)).
                           alternative is the 'exp' function as , exp(x)
            lossfunc: str, default='binary', the model is used as a binary classification problem, 
                            but this foundation is to be ready to implement future option of regression problem
                            will be used as 'MSE' value to calcualte mean squared error.
        Returns:
            loss: objective loss function, 
                  this value will be used by the optimizer to find the w_k (weight of best node), 
                  that best minimizing the function
            grad: gradient of the loss function, 
                    can be used by some of the optimizers to faster and accurate 
                    find the w_k that best minimizing lost function
        '''
        m = y.shape[0] 
        if lossfunc == 'binary':
            farg = 1 - loss_notk - np.dot(h_k,w_k.reshape(-1,1))*y  
            loss = 1/m * self._slfunc(farg,surrloss).sum(axis=0) + reg_notk + np.dot(reg_k,np.abs(w_k))
            if self.optIsGrad:
                grad = -1/m * (self._slgrad(farg,surrloss)*y*h_k).sum(axis=0) + np.dot(np.sign(w_k),reg_k)
                return loss,grad
    #     case 'MSE'
    #         farg = loss_notk + abs(w_k*h_k - y).^2;
    #         loss = 1/m * sum(farg) + reg_notk + reg_k*abs(w_k).^2;
    #         grad = ( -1/m * sum(farg.*h_k) +sign(w_k)*reg_k);
        return loss


    def _slfunc(self,x, func):
        ''' surrogate loss function
        the surrogate loss function is defined in the article 
        as the capital phi, that is activated on the difference of the zero/one loss problem (1-y*(Sig(w*h)))
        this is in order to be sure the sub problem is convex for optimization.
        Args:
            x: Ndarray, can be in different input shapes, will return same shape as input
               the value to activate upon, the surrogate loss function
            func: str, either 'logistic' or 'exp' to activate relevant function on the input value
        Returns:
            val: Ndarray, same shape as input x,
                 the resulted value by the surrogate activation function used on the input value
        '''
        choices = {'logistic': np.log(1+np.exp(x)), 'exp': np.exp(x)}
        val = choices.get(func, x)
        return val


    def _slgrad(self,x, func):
        '''gradient of surrogate loss function
        Args:
            x: Ndarray, can be in different input shapes, will return same shape as input
               the value to activate upon, the gradient of surrogate loss function
            func: str, either 'logistic' or 'exp' to activate relevant gradient function on the input value
        Returns:
            valG: Ndarray, same shape as input x,
                 the resulted value by the gradient surrogate activation function used on the input value
        '''
        choices = {'logistic': np.exp(x)/(1+np.exp(x)), 'exp': np.exp(x)}
        valG = choices.get(func, x)
        return valG

    def _adanet_predict(self,params, Xdata):
        '''inner method of predict implementation of AdaNet
        This method is feeding the input data across the learned AdaNet network, 
        and evaluate for each input the probability of target value on scale -1 to 1.
        Args:
            params: dict, learned architecture and parameters of the network
            Xdata: Ndarray[m_samples,n_features], test samples to evaluate target values
        Returns:
            pred: Ndarray [m_samples,1]
                    probability of target value on scale -1 to 1
        '''       
        if params['augment']:
            Xdata = np.hstack((np.ones((Xdata.shape[0],1)),Xdata))
        if params['augment'] and params['augmentLayers']:
            Xdata = Xdata[:,1:] 

        H = None
        # Compute adanet prediction
        pred = 0
        for k in range(params['numLayers']): 
            if k == 0: 
                Hk = Xdata
                actFunc = 'none'
            else:
                Hk = H
                actFunc = params['activationFunction']

            if params['augmentLayers']:
                Hk = np.hstack((np.ones((Hk.shape[0],1)),Hk))   
                
            H = np.dot(self._activation(Hk,actFunc),params['u'][k])  

            for j in range(params['numNodes'][k,0]): 
                pred = pred + np.dot(H[:,j].reshape(-1,1),params['W'][k][j].reshape(-1,1)) 

            pred = pred + params['W_bias'] 

        return pred

    def fit(self, X, y):
        """ Fit method of the AdaNet classifier
        Args:
            X: Ndarray[m_samples,n_features], train samples to learn the optimal neural network, 
                that is required by this specific problem. complex problems will result deep and wide network,
                while simple ones will result simple architecure of the network
            y: Ndarray[m_samples,1], train target labels in [-1,1], as a binary problem
        """
        if y.ndim == 1:  # Ensure y is 2D, to have matrix multiplications correct
            y = y.reshape(-1,1)
        self.adaParams = {}
        self.history = {}
        self.adaParams,self.history = self._adanet(X,y,self.cfg)

    def predict(self, X):
        '''Predict implementation of AdaNet
        This method calling the inner predict method that 
        is feeding the input data across the learned AdaNet network, 
        and evaluate for each input the target value.
        Args:
            X: Ndarray[m_samples,n_features], test samples to evaluate target values
        Returns:
            y_pred: Ndarray [m_samples,1]
                    target evaluated labels in [-1,1]
        '''
        y_pred = self._adanet_predict(self.adaParams, X)
        if self.adaParams['lossFunction'] == 'binary': 
            y_pred[y_pred>=0] = 1
            y_pred[y_pred<0] = -1
        return y_pred

    def predict_proba(self, X):
        '''Probability estimates of AdaNet
        This method provides the probabilities of predicted lables based on test dataset.
        Args:
            X: Ndarray[m_samples,n_features], test samples to evaluate target values probabilities
        Returns:
            predict_proba(): Ndarray [m_samples,1]
                            based on the binary problem evaluate the positive probability 
                            of target label in range 0 to 1.
        '''
        y_pred = self._adanet_predict(self.adaParams, X)
        if self.adaParams['lossFunction'] == 'binary': 
            y_pred = 0.5 + y_pred/2
        return y_pred

    def score(self, X, y):
        """Score implementation of AdaNet
        used to measure accuracy score (mean) of given test data and labels.
        Args: 
            X,y test data and true labels, in order to predict and compare with true labels,
              measuring the accuracy score
        Returns: 
            score(): float, accuracy score
        """
        return accuracy_score(y,self.predict(X))
