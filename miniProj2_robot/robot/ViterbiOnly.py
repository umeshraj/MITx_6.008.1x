# %% load packaages
import collections
import sys

import graphics
import numpy as np
import robot

# %% local functions
def printProb(inDict):
    print(sorted(inDict.items(), key=lambda x: x[1],
                 reverse=True)[:5])

def rev_transition_model(curState):
    # given a hidden state, return the Distribution for the prev hidden state
    revModel = robot.Distribution()
    for x in all_possible_hidden_states:
        tmp = transition_model(x)
        revModel[x] = tmp[curState]
    #revModel.renormalize()
    return revModel

def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log2(x)

def neglog(x):
    return -1 * careful_log(x)

def get_obs(y_in):
    phiOut = robot.Distribution()
    for x in all_possible_hidden_states:
        y_poss = observation_model(x)
        if y_poss[y_in] > 0:
            phiOut[x] = y_poss[y_in]
    return phiOut


def myDictMin(inDict):
    minVal = np.inf
    minKey = None
    for key, val in inDict.items():
        if val < minVal:
            minVal = val
            minKey = key
    return minVal, minKey

#def myDictMin(inDict):
#    minKey = min(inDict, key=inDict.get)
#    return inDict[minKey], minKey

def myneglog(pDist):
    pDist = pDist.copy()
    pOut = robot.Distribution()
    for key, val in pDist.items():
        pOut[key] = -1*careful_log(val)
    return pOut

def mostLikely(neglog, msgHat):
    finNode = robot.Distribution()
    for key in neglog.keys():
        finNode[key] = neglog[key] + msgHat[key]
    minVal, minKey = myDictMin(finNode)
    mHat = minVal
    tBack = minKey
    return mHat, tBack

# %% main program

# load constants
#all_possible_hidden_states = robot.get_all_hidden_states()
#all_possible_observed_states = robot.get_all_observed_states()
#prior_distribution = robot.initial_distribution()
#transition_model = robot.transition_model
#observation_model = robot.observation_model
#observations = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0),
#                (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]

#import classCoins as cc
#all_possible_hidden_states = cc.get_all_hidden_states()
#all_possible_observed_states = cc.get_all_observed_states()
#prior_distribution = cc.initial_distribution()
#transition_model = cc.transition_model
#observation_model = cc.observation_model
#observations = ['H', 'H', 'T', 'T', 'T']
#g = np.log2(3)

# load wiki examples
import dnaExample as dna
all_possible_hidden_states = dna.get_all_hidden_states()
all_possible_observed_states = dna.get_all_observed_states()
prior_distribution = dna.initial_distribution()
transition_model = dna.transition_model
observation_model = dna.observation_model
observationsStr = 'GGCACTGAA'.lower()
observations = [x for x in observationsStr]

# %% computing m12
#obs1 = get_obs((2, 0))
obs1 = get_obs(observations[0])
prior1 = robot.Distribution()
for x in obs1.keys():
    prior1[x] = prior_distribution[x]
phi1 = robot.Distribution()
for x in obs1.keys():
    tmpProd= obs1[x] * prior1[x]
    if tmpProd > 0:
        phi1[x] = tmpProd

# compute message 1 to 2
m12 = robot.Distribution()
tBack12 = {}
for x2_state in all_possible_hidden_states:
    x1_collect = {}
    for x1_state, x1_value in phi1.items():
        x2_x1_trans = transition_model(x1_state)
        trans_value = x2_x1_trans[x2_state]
        prod = neglog(x1_value) + neglog(trans_value)
        if prod < np.inf:
            x1_collect[x1_state] = prod
    minVal, minKey = myDictMin(x1_collect)
    if minVal < np.inf:
        m12[x2_state] = minVal
        tBack12[x2_state] = minKey


# %% compute message 2 to 3

msgList = [m12]
tBackList = [tBack12]

for idx, y in enumerate(observations[1:-1], start=1):
    prevMsg = msgList[idx-1]
    phi2 = get_obs(y)
    m23 = robot.Distribution()
    tBack23 = {}
    for x2_state in all_possible_hidden_states:
#        if (idx== 3) and x2_state == (4, 0, 'right'):
#            print('debug')
        x1_collect = {}
        for x1_state, x1_value in phi2.items():
            x2_x1_trans = transition_model(x1_state)
            trans_value = x2_x1_trans[x2_state]
            prev = prevMsg[x1_state]
            prod = neglog(x1_value) + neglog(trans_value) + prev
            if prod < np.inf:
                x1_collect[x1_state] = prod

        minVal, minKey = myDictMin(x1_collect)
        if minVal < np.inf:
            m23[x2_state] = minVal
            tBack23[x2_state] = minKey
    msgList.append(m23)
    tBackList.append(tBack23)


# %% just fake the tracke back for now
num_time_steps = len(observations)
fin_phi_neglog = myneglog(get_obs(observations[-1]))
finStates = [None] * num_time_steps
finhat, finState = mostLikely(fin_phi_neglog, msgList[-1])
finStates[-1] = finState
#finStates[-1] = (6, 2, "down")
for idx in range(num_time_steps-1, 0, -1):
    curState = finStates[idx]
    tBack = tBackList[idx-1]
    prevState = tBack[curState]
    finStates[idx-1] = prevState
    print("{0}: {1}".format(idx, curState))
