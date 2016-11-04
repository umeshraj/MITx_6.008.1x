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
    all_states = ['H', 'L']
    return all_states

def get_all_observed_states():
    # lists all possible observed states
    all_observed_states = ['a', 'c', 'g', 't']
    return all_observed_states

def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = robot.Distribution()
    prior['H'] = 0.50
    prior['L'] = 0.50
    prior.renormalize()
    return prior


def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    next_states = robot.Distribution()

    # we can always stay where we are
    if state == 'H':
        next_states['H'] = .50
        next_states['L'] = .50
    elif state == 'L':
        next_states['H'] = .40
        next_states['L'] = .60
    next_states.renormalize()
    return next_states

def observation_model(state):
    observed_states = robot.Distribution()
    if state == 'H':
        observed_states['a'] = 0.2
        observed_states['c'] = 0.3
        observed_states['g'] = 0.3
        observed_states['t'] = 0.2
    elif state == 'L':
        observed_states['a'] = 0.3
        observed_states['c'] = 0.2
        observed_states['g'] = 0.2
        observed_states['t'] = 0.3
    observed_states.renormalize()
    return observed_states
