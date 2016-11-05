#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:50:21 2016
http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf
@author: umesh
"""

import robot

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
    prior = robot.Distribution()
    prior['F'] = 0.50
    prior['B'] = 0.50
    prior.renormalize()
    return prior


def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    next_states = robot.Distribution()

    # we can always stay where we are
    if state == 'F':
        next_states['F'] = .75
        next_states['B'] = .25
    elif state == 'B':
        next_states['F'] = .25
        next_states['B'] = .75
    next_states.renormalize()
    return next_states

def observation_model(state):
    observed_states = robot.Distribution()
    if state == 'F':
        observed_states['H'] = 0.5
        observed_states['T'] = 0.5
    elif state == 'B':
        observed_states['H'] = 0.25
        observed_states['T'] = 0.75
    observed_states.renormalize()
    return observed_states
