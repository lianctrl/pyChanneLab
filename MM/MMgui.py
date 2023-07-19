import tkinter as tk

#class MarkovModelEditor(tk.Frame):
#    def __init__(self, master=None):
#        super().__init__(master)
#        self.master = master
#        self.master.title("Markov Model Editor")
#        self.pack()
#        
#        self.states = []
#        self.transition_probabilities = {}
#        
#        self.canvas = tk.Canvas(self, width=500, height=500, bg='white')
#        self.canvas.pack()
#        
#        self.current_state = None
#        self.canvas.bind('<Button-1>', self.create_state)
#        
#        self.selected_state = None
#        self.canvas.bind('<Button-2>', self.select_state)
#        self.canvas.bind('<Button-3>', self.create_transition)
#        
#    def create_state(self, event):
#        x, y = event.x, event.y
#        state_name = f'State{len(self.states)}'
#        self.canvas.create_oval(x-25, y-25, x+25, y+25, fill='lightblue', tags=state_name)
#        self.canvas.create_text(x, y, text=state_name, tags=state_name)
#        self.add_state(state_name)
#        
#    def add_state(self, state):
#        self.states.append(state)
#        self.transition_probabilities[state] = {}
#        
#    def select_state(self, event):
#        item = self.canvas.find_closest(event.x, event.y)[0]
#        tags = self.canvas.itemcget(item, 'tags')
#        if 'State' in tags:
#            if self.selected_state is not None:
#                self.canvas.itemconfig(self.selected_state, outline='black')
#            self.selected_state = item
#            self.canvas.itemconfig(item, outline='red')
#        
#    def create_transition(self, event):
#        if self.selected_state is not None:
#            x, y = event.x, event.y
#            from_state = self.canvas.itemcget(self.selected_state, 'tags').split()[0]
#            to_state = None
#            for item in self.canvas.find_withtag('State'):
#                if item != self.selected_state:
#                    bbox = self.canvas.bbox(item)
#                    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
#                        to_state = self.canvas.itemcget(item, 'tags').split()[0]
#                        break
#            if to_state is not None:
#                self.canvas.create_line(event.x, event.y, self.canvas.coords(self.selected_state)[0]+25, self.canvas.coords(self.selected_state)[1]+25, arrow='last', tags=(from_state, to_state))
#                self.add_transition(from_state, to_state, 0.5)
#        
#    def add_transition(self, from_state, to_state, probability):
#        self.transition_probabilities[from_state][to_state] = probability
#        
#    def get_transition_probabilities(self):
#        return self.transition_probabilities
#    
#    def clear(self):
#        self.canvas.delete('all')
#        self.states = []
#        self.transition_probabilities = {}
#        self.selected_state = None

import tkinter as tk

class MarkovModelEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Markov Model Editor")
        self.pack()
        
        self.states = []
        self.transition_probabilities = {}
        
        self.canvas = tk.Canvas(self, width=500, height=500, bg='white')
        self.canvas.pack()
        
        # Create buttons for adding states and transitions
        self.add_state_button = tk.Button(self, text='Add State', command=self.create_state)
        self.add_state_button.pack(side='left')
        self.add_transition_button = tk.Button(self, text='Add Transition', command=self.create_transition)
        self.add_transition_button.pack(side='left')
        
        self.current_state = None
        self.canvas.bind('<Button-1>', self.select_state)
        
        self.selected_state = None
        self.selected_transition = None
        
    def create_state(self):
        state_name = f'State{len(self.states)}'
        self.canvas.create_oval(50, 50, 150, 150, fill='lightblue', tags=state_name)
        self.canvas.create_text(100, 100, text=state_name, tags=state_name)
        self.add_state(state_name)
        
    def add_state(self, state):
        self.states.append(state)
        self.transition_probabilities[state] = {}
        
    def select_state(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.itemcget(item, 'tags')
        if 'State' in tags:
            if self.selected_state is not None:
                self.canvas.itemconfig(self.selected_state, outline='black')
            self.selected_state = item
            self.selected_transition = None
            self.canvas.itemconfig(item, outline='red')
        
    def create_transition(self):
        if self.selected_state is not None:
            self.selected_transition = None
            self.canvas.bind('<Button-1>', self.select_transition_start)
            self.canvas.bind('<ButtonRelease-1>', self.select_transition_end)
            self.canvas.bind('<Motion>', self.draw_transition)
        
    def select_transition_start(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.itemcget(item, 'tags')
        if 'State' in tags:
            self.selected_transition = {'from_state': tags.split()[0], 'start_coords': (event.x, event.y)}
        
    def select_transition_end(self, event):
        if self.selected_transition is not None:
            item = self.canvas.find_closest(event.x, event.y)[0]
            tags = self.canvas.itemcget(item, 'tags')
            if 'State' in tags:
                self.add_transition(self.selected_transition['from_state'], tags.split()[0], 0.5)
                self.selected_transition = None
                self.canvas.unbind('<Button-1>')
                self.canvas.unbind('<ButtonRelease-1>')
                self.canvas.unbind('<Motion>')
        
    def draw_transition(self, event):
        if self.selected_transition is not None:
            coords = self.selected_transition['start_coords'] + (event.x, event.y)
            if self.selected_transition.get('line') is None:
                self.selected_transition['line'] = self.canvas.create_line(*coords, arrow='last')
            else:
                self.canvas.coords(self.selected_transition['line'], *coords)
        
    def add_transition(self, from_state, to_state, probability):
        self.transition_probabilities[from_state][to_state] = probability

    def get_transition_probabilities(self):
        return self.transition_probabilities


root = tk.Tk()
app = MarkovModelEditor(root)
root.mainloop()




