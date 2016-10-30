#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:59:27 2016

@author: umesh
"""
from robot import Distribution




def get_all_hidden_states():
    # lists all possible hidden states
    all_states = ['1', '2', '3']
    return all_states


def get_all_observed_states():
    # lists all possible observed states
    all_observed_states = ['hot', 'cold']
    return all_observed_states


def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = Distribution()
    prior['1'] = 1
    prior['2'] = 1
    prior['3'] = 1
    prior.renormalize()
    return prior

def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    next_states = Distribution()

    # we can always stay where we are
    if state == '1':
        next_states['1'] = .25
        next_states['2'] = .75
        next_states['3'] = 0
    elif state == '2': 
        next_states['1'] = 0
        next_states['2'] = .25
        next_states['3'] = .75
    elif state == '3': 
        next_states['1'] = 0
        next_states['2'] = 0
        next_states['3'] = 1

    next_states.renormalize()
    return next_states
    
def observation_model(state):
    observed_states = Distribution()
    if state == '1':
        observed_states['hot'] = 1
        observed_states['cold'] = 0
    elif state == '2':
        observed_states['hot'] = 0
        observed_states['cold'] = 1
    elif state == '3':
        observed_states['hot'] = 1
        observed_states['cold'] = 0
    observed_states.renormalize()
    return observed_states
    
    
def buildPhi(y):
    phi_X = Distribution()
    for x in all_possible_hidden_states:
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
    
    
all_possible_hidden_states = get_all_hidden_states()
all_possible_observed_states = get_all_observed_states()
prior_distribution = initial_distribution()
transition_model = transition_model
observation_model = observation_model    
if __name__ == '__main__':
    observations = ['hot', 'cold', 'hot']
    forward_backward(observations)