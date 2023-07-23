class MarkovModel:
    def __init__(self):
        self.states = {}
        self.transitions = {}

    def add_state(self, state_name):
        if state_name not in self.states:
            self.states[state_name] = {}

    def add_transition(self, from_state, to_state, rate_function):
        if from_state not in self.states:
            self.add_state(from_state)
        if to_state not in self.states:
            self.add_state(to_state)

        self.transitions[(from_state, to_state)] = rate_function

    def get_transition_rate(self, from_state, to_state):
        return self.transitions.get((from_state, to_state), None)
