#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:59:38 2016

@author: umesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:59:27 2016

@author: umesh
"""
from robot import Distribution
import numpy as np

# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log2(x)


def get_all_hidden_states():
    # lists all possible hidden states
    all_states = ['F', 'B']
    return all_states

def get_all_observed_states():
    # lists all possible observed states
    all_observed_states = ['H', 'T']
    return all_observed_states


def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = Distribution()
    prior['F'] = 1
    prior['B'] = 1
    prior.renormalize()
    return prior

def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    next_states = Distribution()

    # we can always stay where we are
    if state == 'F':
        next_states['F'] = .75
        next_states['B'] = .25
    elif state == 'B': 
        next_states['F'] = 0.25
        next_states['B'] = 0.75
    next_states.renormalize()
    return next_states
    
def observation_model(state):
    observed_states = Distribution()
    if state == 'F':
        observed_states['H'] = .5
        observed_states['T'] = 0.5
    elif state == 'B':
        observed_states['H'] = .25
        observed_states['T'] = .75
    observed_states.renormalize()
    return observed_states
    
    
def buildPhi(y):
    phi_X = Distribution()
    for x in all_possible_hidden_states:
        if y is None:
            phi_X[x] = 1
        else:
            yPoss = observation_model(x)
            phi_X[x] = yPoss[y]                
    return phi_X

def forward(alphaIn, phi_x, y):
    """compute the next forward message"""
    alphaPhi_X = Distribution()
    for x, alphaX in alphaIn.items():
        yProb = phi_x[x]
        tmpProd = yProb * alphaX
        if tmpProd > 0:
            alphaPhi_X[x] = tmpProd
    
    # compute alpha out
    alphaOut = Distribution()
    for x, alphaPhi in alphaPhi_X.items():
        x2Poss = transition_model(x)
        # multiply and add x2Poss to o/p
        for x2Key, x2pVal in x2Poss.items():
            alphaOut[x2Key] += x2pVal*alphaPhi
        #print(alphaOut)
    return alphaOut         

def rev_transition_model(curState):
    # given a hidden state, return the Distribution for the prev hidden state
    revModel = Distribution()
    for x in all_possible_hidden_states:
        tmp = transition_model(x)
        revModel[x] = tmp[curState]
    return revModel

    
def backward(alphaIn, phi_x, y):
    """compute the next forward message"""
    alphaPhi_X = Distribution()
    for x, alphaX in alphaIn.items():
        yProb = phi_x[x]
        tmpProd = yProb * alphaX
        if tmpProd > 0:
            alphaPhi_X[x] = tmpProd
    
    # compute alpha out
    alphaOut = Distribution()
    for x, alphaPhi in alphaPhi_X.items():
        x2Poss = rev_transition_model(x)
        # multiply and add x2Poss to o/p
        for x2Key, x2pVal in x2Poss.items():
            alphaOut[x2Key] += x2pVal*alphaPhi
        #print(alphaOut)
    return alphaOut    
    
def mkMarginals(fwd, back, phi):
    marg = Distribution()
    for x in all_possible_hidden_states:
        marg[x] = phi[x] * fwd[x] * back[x]
    marg.renormalize()
    return marg
    
def printProb(inDict):
    print(sorted(inDict.items(), key=lambda x: x[1], 
                 reverse=True)[:10])
# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    observations = ['H', 'H', 'T', 'T', 'T']
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution

    # pre-build phi_x for all nodes
    phi_XList = [None] * num_time_steps
    for idx, y in enumerate(observations):
        phi_XList[idx] = buildPhi(y)

    # TODO: Compute the forward messages
    for idx, y in enumerate(observations[0:-1]):
        alphaIn = forward_messages[idx]
        nxtFwd = forward(alphaIn, phi_XList[idx], y)
        forward_messages[idx+1] = nxtFwd

    backward_messages = [None] * num_time_steps
    # TODO: Compute the backward messages
    backMsg = Distribution()
    for x in all_possible_hidden_states:
        backMsg[x] = 1
    backward_messages[-1] = backMsg
    for idx, y in enumerate(observations[-1:0:-1]):
        nodeIdx = num_time_steps-idx-1
        if nodeIdx == 99:
            print('Enter debug')
            print('Enter debug')
        betaIn = backward_messages[nodeIdx]
        nxtBeta = backward(betaIn, phi_XList[nodeIdx], y)
        backward_messages[nodeIdx-1] = nxtBeta
 
    testIdx = 1
    fwdTest = forward_messages[testIdx]
    fwdTest.renormalize()
    printProb(fwdTest)
    backTest = backward_messages[testIdx]
    backTest.renormalize()
    printProb(backTest)

    #marginals = [None] * num_time_steps # remove this
    marginals = []
    # TODO: Compute the marginals 
    fbpZip = zip(forward_messages, backward_messages, phi_XList)
    for fwd, back, phi in fbpZip:        
        marg = mkMarginals(fwd, back, phi)
        marginals.append(marg)
    return marginals
    
def myneglog(pDist):
    pDist = pDist.copy()
    for key, val in pDist.items():
        pDist[key] = -1*careful_log(val)
    return pDist
    
def myDictMin(inDict):
    minKey = min(inDict, key=inDict.get)
    return inDict[minKey], minKey
    
def ViterbiWkHorse(neglogphi, prevMsgHat):
    mHat = Distribution()
    for x2Key in all_possible_hidden_states:
        tmp = Distribution()
        for x1Key, x1Val in neglogphi.items():
            x2Poss = transition_model(x1Key)
            neglogx2Poss = myneglog(x2Poss)
            tmp[x1Key] = x1Val + neglogx2Poss[x2Key] + prevMsgHat[x1Key]
        minVal, minKey = myDictMin(tmp)
        mHat[x2Key] = minVal
    return mHat

def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    observations = ['H', 'H', 'T', 'T', 'T']
    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this
    
    # pre-build phi_x for all nodes
    phi_XList = [None] * num_time_steps
    for idx, y in enumerate(observations):
        phi_XList[idx] = buildPhi(y)
    # change phi for first observation
    phi_XList[0]['F'] *= prior_distribution['F']
    phi_XList[0]['B'] *= prior_distribution['B']

    #mHatInit = 
    #for idx, y in enumerate(observations)
    #y = observations[0]
    neglogphi = myneglog(phi_XList[0])
    mHatZero = Distribution()
    for x in all_possible_hidden_states:
        mHatZero[x] = 0
    mHat12 = ViterbiWkHorse(neglogphi, mHatZero)
    
    neglogphi = myneglog(phi_XList[1])
    mHat23 = ViterbiWkHorse(neglogphi, mHat12)

    print(mHat)
    return estimated_hidden_states    
    
all_possible_hidden_states = get_all_hidden_states()
all_possible_observed_states = get_all_observed_states()
prior_distribution = initial_distribution()
transition_model = transition_model
observation_model = observation_model    
if __name__ == '__main__':
    observations = ['hot', 'cold', 'hot']
    #forward_backward(observations)
    Viterbi(None)