'''
import random

class MarkovModel:
    def __init__(self, states, transition_probabilities):
        self.states = states
        self.transition_probabilities = transition_probabilities

    def next_state(self, current_state):
        next_state = None
        transition_probabilities = self.transition_probabilities[current_state]
        random_value = random.random()
        cumulative_probability = 0.0
        for state, probability in transition_probabilities.items():
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                next_state = state
                break
        return next_state

states = ['A', 'B', 'C']
transition_probabilities = {
   'A': {'A': 0.4, 'B': 0.35, 'C': 0.15},
   'B': {'A': 0.20, 'B': 0.5, 'C': 0.30},
   'C': {'A': 0.25, 'B': 0.25, 'C': 0.5}
}

markov_model = MarkovModel(states, transition_probabilities)
current_state = 'A'
for i in range(10):
    next_state = markov_model.next_state(current_state)
    print(f"Step {i}: {current_state} -> {next_state}")
    current_state = next_state
'''

class MarkovModel:
    def __init__(self):
        self.states = []
        self.transition_probabilities = {}

    def add_state(self, state):
        self.states.append(state)
        self.transition_probabilities[state] = {}

    def add_transition(self, from_state, to_state, probability):
        self.transition_probabilities[from_state][to_state] = probability

    def next_state(self, current_state):
        import random

        next_state = None
        transition_probabilities = self.transition_probabilities[current_state]
        random_value = random.random()
        cumulative_probability = 0.0
        for state, probability in transition_probabilities.items():
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                next_state = state
                break
        return next_state

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


markov_model = MarkovModel()

# Add states
markov_model.add_state('A')
markov_model.add_state('B')
markov_model.add_state('C')

# Add transitions
markov_model.add_transition('A', 'A', 0.9)
markov_model.add_transition('A', 'B', 0.075)
markov_model.add_transition('A', 'C', 0.025)
markov_model.add_transition('B', 'A', 0.15)
markov_model.add_transition('B', 'B', 0.8)
markov_model.add_transition('B', 'C', 0.05)
markov_model.add_transition('C', 'A', 0.25)
markov_model.add_transition('C', 'B', 0.25)
markov_model.add_transition('C', 'C', 0.5)

current_state = 'A'
for i in range(10):
    next_state = markov_model.next_state(current_state)
    print(f"Step {i}: {current_state} -> {next_state}")
    current_state = next_state
