#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)

def buildPhi(y):
    phi_X = robot.Distribution()
    for x in all_possible_hidden_states:
        if y is None:
            phi_X[x] = 1
        else:
            yPoss = observation_model(x)
            phi_X[x] = yPoss[y]
    return phi_X

def forward(alphaIn, phi_x, y):
    """compute the next forward message"""
    alphaPhi_X = robot.Distribution()
    for x, alphaX in alphaIn.items():
        yProb = phi_x[x]
        tmpProd = yProb * alphaX
        if tmpProd > 0:
            alphaPhi_X[x] = tmpProd

    # compute alpha out
    alphaOut = robot.Distribution()
    for x, alphaPhi in alphaPhi_X.items():
        x2Poss = transition_model(x)
        # multiply and add x2Poss to o/p
        for x2Key, x2pVal in x2Poss.items():
            alphaOut[x2Key] += x2pVal*alphaPhi
        #print(alphaOut)
    return alphaOut

def rev_transition_model(curState):
    # given a hidden state, return the Distribution for the prev hidden state
    revModel = robot.Distribution()
    for x in all_possible_hidden_states:
        tmp = transition_model(x)
        revModel[x] = tmp[curState]
    #revModel.renormalize()
    return revModel

def ur_rev_transition_model():
    ur_rev_trans_dict = {}
    for x1 in all_possible_hidden_states:
        ur_rev_trans_dict[x1] = rev_transition_model(x1)
    return ur_rev_trans_dict

ur_rev_trans_dict = ur_rev_transition_model()


def backward(alphaIn, phi_x, y):
    """compute the next forward message"""
    alphaPhi_X = robot.Distribution()
    for x, alphaX in alphaIn.items():
        yProb = phi_x[x]
        tmpProd = yProb * alphaX
        if tmpProd > 0:
            alphaPhi_X[x] = tmpProd

    # compute alpha out
    alphaOut = robot.Distribution()
    for x, alphaPhi in alphaPhi_X.items():
        x2Poss = ur_rev_trans_dict[x]
        #x2Poss = rev_transition_model(x)
        # multiply and add x2Poss to o/p
        for x2Key, x2pVal in x2Poss.items():
            alphaOut[x2Key] += x2pVal*alphaPhi
        #print(alphaOut)
    return alphaOut

def mkMarginals(fwd, back, phi):
    marg = robot.Distribution()
    for x in all_possible_hidden_states:
        marg[x] = phi[x] * fwd[x] * back[x]
    marg.renormalize()
    return marg

def printProb(inDict):
    print(sorted(inDict.items(), key=lambda x: x[1],
                 reverse=True)[:2])
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
    backMsg = robot.Distribution()
    for x in all_possible_hidden_states:
        backMsg[x] = 1
    backward_messages[-1] = backMsg
    for idx, y in enumerate(observations[-1:0:-1]):
        nodeIdx = num_time_steps-idx-1
        betaIn = backward_messages[nodeIdx]
        nxtBeta = backward(betaIn, phi_XList[nodeIdx], y)
        backward_messages[nodeIdx-1] = nxtBeta

#    testIdx = 2
#    fwdTest = forward_messages[testIdx]
#    fwdTest.renormalize()
#    printProb(fwdTest)
#    backTest = backward_messages[testIdx]
#    backTest.renormalize()
#    printProb(backTest)

    #marginals = [None] * num_time_steps # remove this
    marginals = []
    # TODO: Compute the marginals
    fbpZip = zip(forward_messages, backward_messages, phi_XList)
    for fwd, back, phi in fbpZip:
        marg = mkMarginals(fwd, back, phi)
        marginals.append(marg)
        # printProb(marg)
    return marginals

def neglog(x):
    return -1 * careful_log(x)

def myDictMin(inDict):
    minVal = np.inf
    minKey = None
    for key, val in inDict.items():
        if val <= minVal:
            minVal = val
            minKey = key
    return minVal, minKey

def myDictMinSec(inDict):
    tmpDict = inDict.copy()
    minVal, minKey = myDictMin(tmpDict)
    tmpDict.pop(minKey)
    minVal, minKey = myDictMin(tmpDict)
    return minVal, minKey

#def myneglog(pDist):
#    pDist = pDist.copy()
#    pOut = robot.Distribution()
#    for key, val in pDist.items():
#        pOut[key] = -1*careful_log(val)
#    return pOut

def myneglog(pDist):
    pOut = {}
    for key, val in pDist.items():
        pOut[key] = -1*careful_log(val)
    return pOut


def mostLikely(phiLast, msgHat):
    finNode = robot.Distribution()
    for key in phiLast.keys():
        val2 = msgHat[key]
        if val2 == 0:
            val2 = np.inf
        tmpSum = neglog(phiLast[key]) + val2
        if tmpSum < np.inf:
            finNode[key] = tmpSum
    minVal, minKey = myDictMin(finNode)
    mHat = minVal
    tBack = minKey
    return mHat, tBack

def mostLikelySec(phiLast, msgHat):
    finNode = robot.Distribution()
    for key in phiLast.keys():
        val2 = msgHat[key]
        if val2 == 0:
            val2 = np.inf
        tmpSum = neglog(phiLast[key]) + val2
        if tmpSum < np.inf:
            finNode[key] = tmpSum
    minVal, minKey = myDictMinSec(finNode)
    mHat = minVal
    tBack = minKey
    return mHat, tBack

def get_all_poss_x2(x1_state_list):
    all_x2 = []
    for x1_state in x1_state_list:
        x2_x1_trans = transition_model(x1_state)
        for x2 in x2_x1_trans:
            all_x2.append(x2)
    all_x2 = set(all_x2)
    return all_x2

def ur_transition_model():
    ur_trans_dict = {}
    for x1 in all_possible_hidden_states:
        #ur_trans_dict[x1] = myneglog(transition_model(x1))
        ur_trans_dict[x1] = transition_model(x1)
    return ur_trans_dict


def ViterbiWkHorse(observations):
    num_time_steps = len(observations)
    #estimated_hidden_states = [None] * num_time_steps # remove this

    # pre-compute for speed
    ur_trans_dict = ur_transition_model()

    # %% computing m12
    phi1 = robot.Distribution()
    obs1 = buildPhi(observations[0])
    for x in obs1.keys():
        tmpProd= obs1[x] * prior_distribution[x]
        if tmpProd > 0:
            phi1[x] = tmpProd

    # compute message 1 to 2
    m12 = robot.Distribution()
    tBack12 = {}
    phi_use = myneglog(phi1)
    all_x1 = list(phi_use.keys())
    #for x2_state in all_possible_hidden_states:
    all_poss_x2 = get_all_poss_x2(all_x1)
    for x2_state in all_poss_x2:
        x1_collect = {}
        for x1_state, x1_value in phi_use.items():
            #x2_x1_trans = transition_model(x1_state)
            x2_x1_trans = ur_trans_dict[x1_state]
            trans_value = x2_x1_trans[x2_state]
            #prod = x1_value + trans_value
            prod = x1_value + neglog(trans_value)
            if prod < np.inf:
                x1_collect[x1_state] = prod
        if bool(x1_collect):
            minVal, minKey = myDictMin(x1_collect)
            if minVal < np.inf:
                m12[x2_state] = minVal
                tBack12[x2_state] = minKey

    # %% compute message 2 to 3

    msgList = [m12]
    prevMsg = m12
    tBackList = [tBack12]
    startIdx = 2
    for idx in range(startIdx, num_time_steps):
        y = observations[idx-1]
        phi2 = myneglog(buildPhi(y))
        #prevMsg = msgList[idx-2]
        m23 = robot.Distribution()
        tBack23 = {}
        all_x1 = list(phi2.keys())
        all_poss_x2 = get_all_poss_x2(all_x1)
        #for x2_state in all_possible_hidden_states:
        for x2_state in all_poss_x2:
            x1_collect = {}
            for x1_state, x1_value in phi2.items():
                #x2_x1_trans = transition_model(x1_state)
                x2_x1_trans = ur_trans_dict[x1_state]
                trans_value = x2_x1_trans[x2_state]
                prev = prevMsg[x1_state]
                if prev == 0:
                    prev = np.inf
                #prod = x1_value + trans_value + prev
                prod = x1_value + neglog(trans_value) + prev
                if prod < np.inf:
                    x1_collect[x1_state] = prod

            if bool(x1_collect):
                minVal, minKey = myDictMin(x1_collect)
                if minVal < np.inf:
                    m23[x2_state] = minVal
                    tBack23[x2_state] = minKey
        msgList.append(m23)
        prevMsg = m23
        tBackList.append(tBack23)
    return tBackList, msgList


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
    num_time_steps = len(observations)
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    tBackList, msgList = ViterbiWkHorse(observations)
    prevMsg = msgList[-1]
    # %% just fake the tracke back for now
    finStates = [None] * num_time_steps

    phiLast = buildPhi(observations[-1])
    #finhat, finState = mostLikely(phiLast, msgList[-1])
    finhat, finState = mostLikely(phiLast, prevMsg)
    finStates[-1] = finState
    #finStates[-1] = (6, 2, "down")
    curState = finStates[-1]
    for idx in range(num_time_steps-1, 0, -1):
        curState = finStates[idx]
        tBack = tBackList[idx-1]
        prevState = tBack[curState]
        finStates[idx-1] = prevState
        # print("{0}: {1}".format(idx, curState))
    # first state update
    firstState= tBack[curState]
    finStates[0] = firstState

    estimated_hidden_states = finStates
    return estimated_hidden_states


def second_best(observations):
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


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

#    tBackList, msgList = ViterbiWkHorse(observations)
#    prevMsg = msgList[-1]
#    # %% just fake the tracke back for now
#    finStates = [None] * num_time_steps
#
#    phiLast = buildPhi(observations[-1])
#    #finhat, finState = mostLikely(phiLast, msgList[-1])
#    finhat, finState = mostLikelySec(phiLast, prevMsg)
#    finStates[-1] = finState
#    finStates[-1] = (11, 2, "down")
#    curState = finStates[-1]
#    for idx in range(num_time_steps-1, 0, -1):
#        curState = finStates[idx]
#        tBack = tBackList[idx-1]
#        prevState = tBack[curState]
#        finStates[idx-1] = prevState
#        # print("{0}: {1}".format(idx, curState))
#    # first state update
#    firstState= tBack[curState]
#    finStates[0] = firstState

    estimated_hidden_states = finStates
    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    # initial_state = prior_distribution.sample()
    initial_state = (0, 0, 'stay')
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = False
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing,
                          random_seed=0)

#    print('Running forward-backward...')
#    marginals = forward_backward(observations)
#    print("\n")
#
#    timestep = 2
#    print("Most likely parts of marginal at time %d:" % (timestep))
#    if marginals[timestep] is not None:
#        print(sorted(marginals[timestep].items(),
#                     key=lambda x: x[1],
#                     reverse=True)[:10])
#    else:
#        print('*No marginal computed*')
#    print("\n")

    print('Forward backward one')
    marg1 = forward_backward([(4, 3), (4, 2), (3, 2), (4, 0), (2, 0),
                      (2, 0), (3, 2), (4, 2), (2, 3), (3, 5)])
    print(marg1)
    print('Forward backward two')
    marg2 = forward_backward([(5, 0), (3, 0), (3, 0), (2, 0), (0, 0),
                              (0, 1), (0, 1), (1, 2), (0, 3), (0, 4)])

    print('Forward backward three')
    forward_backward([(6, 0), (6, 2), (7, 2), (7, 3), (7, 4), (7, 5),
                      (6, 5), (5, 6), None, (7, 7)])

    print('Forward backward four')
    forward_backward([(6, 5), (7, 4), (8, 4), (10, 4), (10, 5), None,
                      (11, 5), (11, 5), (9, 4), None])

    print('Running Viterbi...')
    observations = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0), (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]
    #observations = [(1, 6), (4, 6), (4, 7), None, (5, 6), (6, 5), (6, 6), None, (5, 5), (4, 4)]
    estimated_states = Viterbi(observations)
    print(estimated_states)
    print("\n")

#    print("Last 10 hidden states in the MAP estimate:")
#    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
#        if estimated_states[time_step] is None:
#            print('Missing')
#        else:
#            print(estimated_states[time_step])
#    print("\n")
#
#    print('Finding second-best MAP estimate...')
#    observations = [(8, 2), (8, 1), (10, 0), (10, 0), (10, 1), (11, 0),
#                    (11, 0), (11, 1), (11, 2), (11, 2)]
#    estimated_states2 = second_best(observations)
#    print(estimated_states2)
#    print("\n")
#
#    print("Last 10 hidden states in the second-best MAP estimate:")
#    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
#        if estimated_states2[time_step] is None:
#            print('Missing')
#        else:
#            print(estimated_states2[time_step])
#    print("\n")
#
#    difference = 0
#    difference_time_steps = []
#    for time_step in range(num_time_steps):
#        if estimated_states[time_step] != hidden_states[time_step]:
#            difference += 1
#            difference_time_steps.append(time_step)
#    print("Number of differences between MAP estimate and true hidden " +
#          "states:", difference)
#    if difference > 0:
#        print("Differences are at the following time steps: " +
#              ", ".join(["%d" % time_step
#                         for time_step in difference_time_steps]))
#    print("\n")
#
#    difference = 0
#    difference_time_steps = []
#    for time_step in range(num_time_steps):
#        if estimated_states2[time_step] != hidden_states[time_step]:
#            difference += 1
#            difference_time_steps.append(time_step)
#    print("Number of differences between second-best MAP estimate and " +
#          "true hidden states:", difference)
#    if difference > 0:
#        print("Differences are at the following time steps: " +
#              ", ".join(["%d" % time_step
#                         for time_step in difference_time_steps]))
#    print("\n")
#
#    difference = 0
#    difference_time_steps = []
#    for time_step in range(num_time_steps):
#        if estimated_states[time_step] != estimated_states2[time_step]:
#            difference += 1
#            difference_time_steps.append(time_step)
#    print("Number of differences between MAP and second-best MAP " +
#          "estimates:", difference)
#    if difference > 0:
#        print("Differences are at the following time steps: " +
#              ", ".join(["%d" % time_step
#                         for time_step in difference_time_steps]))
#    print("\n")
#
#    # display
#    if use_graphics:
#        app = graphics.playback_positions(hidden_states,
#                                          observations,
#                                          estimated_states,
#                                          marginals)
#        app.mainloop()


if __name__ == '__main__':
    main()
