# import numpy as np

class MarkovModel:
    def __init__(self):
        self.states = []
        self.transition_probabilities = {}

    def add_state(self, state):
        self.states.append(state)
        self.transition_probabilities[state] = {}

    def add_transition(self, from_state, to_state, probability):
        self.transition_probabilities[from_state][to_state] = probability

    def get_transition_probabilities(self):
        return self.transition_probabilities
