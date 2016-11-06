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
    all_states = ['Healthy', 'Fever']
    return all_states

def get_all_observed_states():
    # lists all possible observed states
    all_observed_states = ['normal', 'cold', 'dizzy']
    return all_observed_states

def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = robot.Distribution()
    prior['Healthy'] = 0.60
    prior['Fever'] = 0.40
    prior.renormalize()
    return prior


def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    next_states = robot.Distribution()

    # we can always stay where we are
    if state == 'Healthy':
        next_states['Healthy'] = .7
        next_states['Fever'] = .3
    elif state == 'Fever':
        next_states['Healthy'] = .4
        next_states['Fever'] = .6
    next_states.renormalize()
    return next_states

def observation_model(state):
    observed_states = robot.Distribution()
    if state == 'Healthy':
        observed_states['normal'] = 0.5
        observed_states['cold'] = 0.4
        observed_states['dizzy'] = 0.1
    elif state == 'Fever':
        observed_states['normal'] = 0.1
        observed_states['cold'] = 0.3
        observed_states['dizzy'] = 0.6
    observed_states.renormalize()
    return observed_states
