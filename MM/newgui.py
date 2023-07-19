import tkinter as tk
from marmod import MarkovModel


class MarkovModelEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Markov Model Editor")
        self.pack()

        self.markov_model = MarkovModel()

        self.canvas = tk.Canvas(self, width=500, height=500, bg='white')
        self.canvas.pack()

        self.current_state = None
        self.canvas.bind('<Button-1>', self.create_state)

        self.selected_state = None
        self.canvas.bind('<Button-2>', self.select_state)
        self.canvas.bind('<Button-3>', self.create_transition)

        self.add_state_button = tk.Button(self, text="Add State", command=self.add_state)
        self.add_state_button.pack(side=tk.LEFT)

        self.add_transition_button = tk.Button(self, text="Add Transition", command=self.add_transition)
        self.add_transition_button.pack(side=tk.LEFT)

    def create_state(self, event):
        x, y = event.x, event.y
        state_name = f'State{len(self.markov_model.states)}'
        self.canvas.create_oval(x - 25, y - 25, x + 25, y + 25, fill='lightblue', tags=state_name)
        self.canvas.create_text(x, y, text=state_name, tags=state_name)
        self.markov_model.add_state(state_name)

    def select_state(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.itemcget(item, 'tags')
        if 'State' in tags:
            if self.selected_state is not None:
                self.canvas.itemconfig(self.selected_state, outline='black')
            self.selected_state = item
            self.canvas.itemconfig(item, outline='red')

    def create_transition(self, event):
        if self.selected_state is not None:
            x, y = event.x, event.y
            from_state = self.canvas.itemcget(self.selected_state, 'tags').split()[0]
            to_state = None
            for item in self.canvas.find_withtag('State'):
                if item != self.selected_state:
                    bbox = self.canvas.bbox(item)
                    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                        to_state = self.canvas.itemcget(item, 'tags').split()[0]
                        break
            if to_state is not None:
                self.canvas.create_line(event.x, event.y, self.canvas.coords(self.selected_state)[0] + 25,
                                        self.canvas.coords(self.selected_state)[1] + 25, arrow='last',
                                        tags=(from_state, to_state))
                self.markov_model.add_transition(from_state, to_state, 0.5)

    def add_state(self):
        self.create_state(None)

    def add_transition(self):
        self.create_transition(None)

#    def get_transition_probabilities(self):
#        return self.markov_model.get_transition_probabilities

root = tk.Tk()
app = MarkovModelEditor(root)
root.mainloop()
