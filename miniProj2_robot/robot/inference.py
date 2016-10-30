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
        x2Poss = rev_transition_model(x)
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
    return marginals


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


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

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
        num_time_steps = 3
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing,
                          random_seed=0)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")
#
#    print("Last 10 hidden states in the MAP estimate:")
#    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
#        if estimated_states[time_step] is None:
#            print('Missing')
#        else:
#            print(estimated_states[time_step])
#    print("\n")
#
#    print('Finding second-best MAP estimate...')
#    estimated_states2 = second_best(observations)
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

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
