''' Module where the MSM is built

It defines the number of states and the possible transitions between them
Define a method that return a matrix of coefficients NxN where N are the
user defined states
'''

#from ...Software.Functions.functions import linear

#print('Hello World')


class MarkovModel:
    def __init__(self):
        self.states = []
        self.transition_functions = {}

    def add_state(self, state):
        self.states.append(state)
        self.transition_functions[state] = {}

    def add_transition(self, from_state, to_state, transition_probability_function):
        self.transition_functions[from_state][to_state] = transition_probability_function

    def next_state(self, current_state, time):
        import random

        next_state = None
        transition_functions = self.transition_functions[current_state]
        random_value = random.random()
        cumulative_probability = 0.0
        for state, transition_probability_function in transition_functions.items():
            probability = transition_probability_function(time)
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                next_state = state
                break
        return next_state
