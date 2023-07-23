import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog as simpledialog
from tkinter.filedialog import asksaveasfilename, askopenfilename
import math
from numpy.random import rand
import pickle as pkl


class State:
    def __init__(self, canvas, x, y, name):
        self.canvas = canvas
        self.item = canvas.create_oval(x-20, y-20, x+20, y+20,
                                       fill='lightblue', outline='black')
        self.canvas.tag_bind(self.item, '<Button1-Motion>', self.move)
        self.x = x
        self.y = y
        self.arrows = []
        self.name = name
        self.label = canvas.create_text(x, y+30, text=name)

    def move(self, event):
        dx = event.x - self.x
        dy = event.y - self.y
        self.canvas.move(self.item, dx, dy)
        self.canvas.move(self.label, dx, dy)
        self.x = event.x
        self.y = event.y
        for arrow in self.arrows:
            arrow.redraw()


class Transition:
    def __init__(self, canvas, start, end, name):
        self.canvas = canvas
        self.start = start
        self.end = end
        self.rate_function = None
        self.line = canvas.create_line(start.x, start.y, end.x, end.y,
                                       arrow='last')
        start.arrows.append(self)
        end.arrows.append(self)
        self.name = name
        self.label = canvas.create_text((start.x+end.x)/2, (start.y+end.y)/2,
                                        text=name, fill="red")

    def redraw(self):
        self.canvas.coords(self.line, self.start.x, self.start.y, self.end.x,
                           self.end.y)
        angle = math.atan2(self.end.y - self.start.y,
                           self.end.x - self.start.x)
        label_x = (self.start.x + self.end.x) / 2 + 40 * math.cos(angle)
        label_y = (self.start.y + self.end.y) / 2 + 40 * math.sin(angle)
        self.canvas.coords(self.label, label_x, label_y)

    def set_rate_function(self, rate_function):
        self.rate_function = rate_function
        self.name = self.name
        if rate_function is not None:
            self.canvas.itemconfig(self.label, text=self.name, fill='black')
        else:
            self.canvas.itemconfig(self.label, text=self.name, fill='red')


class MarkovModel:
    def __init__(self):
        self.states = {}
        self.transitions = []

    def add_state(self, state_name):
        if state_name not in self.states:
            self.states[state_name] = len(self.states)

    def add_transition(self, tname, from_state, to_state, rate_function):
        from_state_idx = self.states[from_state]
        to_state_idx = self.states[to_state]
        self.transitions.append((tname, from_state_idx, to_state_idx,
                                 rate_function))


class MMeditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Markov Model Editor")
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.circles = {}
        self.arrows = []

        self.add_circle_button = tk.Button(master, text='Add State',
                                           command=self.add_circle)
        self.add_circle_button.pack(side=tk.LEFT)

        self.add_arrow_button = tk.Button(master, text='Add Transition',
                                          command=self.ask_for_transition)
        self.add_arrow_button.pack(side=tk.LEFT)

        self.add_rate_button = tk.Button(master, text='Add Rate Function',
                                         command=self.ask_for_rate)
        self.add_rate_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(master, text='Clear All',
                                      command=self.clear)
        self.clear_button.pack(side=tk.LEFT)

        self.table_button = tk.Button(master, text='Show Table',
                                      command=self.show_table)
        self.table_button.pack(side=tk.LEFT)

        self.table_frame = tk.Frame(master)
        self.table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.table_tree = ttk.Treeview(self.table_frame)
        self.table_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.table_scroll = ttk.Scrollbar(self.table_frame,
                                          orient=tk.VERTICAL,
                                          command=self.table_tree.yview)
        self.table_tree.configure(yscroll=self.table_scroll.set)
        self.table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.export_button = tk.Button(master, text='Export Model',
                                       command=self.export_model)
        self.export_button.pack(side=tk.LEFT)

        self.import_button = tk.Button(master, text='Import Model',
                                       command=self.import_model)
        self.import_button.pack(side=tk.LEFT)

    def add_circle(self):
        name = simpledialog.askstring('State Name', 'Enter a name'
                                      ' for the State:')
        if not name:
            return
        x, y = rand()*250, rand()*250
        states = State(self.canvas, x, y, name)
        self.circles[name] = states

    def ask_for_transition(self):
        if len(self.circles) < 2:
            return
        start_name = simpledialog.askstring('Start State', 'Enter the name of'
                                            ' the starting state:')
        if not start_name or start_name not in self.circles:
            return
        end_name = simpledialog.askstring('End State', 'Enter the name of '
                                          'the ending state:')
        if not end_name or end_name not in self.circles or\
           end_name == start_name:
            return
        name = simpledialog.askstring('Transition Name', 'Enter name for'
                                      ' the transition:')
        if not name:
            return

        def create_transition():
            start = self.circles[start_name]
            end = self.circles[end_name]
            arrow = Transition(self.canvas, start, end, name)
            self.arrows.append(arrow)
            if reverse_var.get():
                arrow_reverse = Transition(self.canvas, end, start, name +
                                           'bw')
                self.arrows.append(arrow_reverse)

            transition_window.destroy()

        transition_window = tk.Toplevel(self.master)
        transition_window.title("Add Transition")
        reverse_var = tk.BooleanVar()
        reverse_var.set(False)

        tk.Label(transition_window, text="Create a reverse transition?").pack()
        tk.Checkbutton(transition_window, text="Yes",
                       variable=reverse_var).pack()
        tk.Button(transition_window, text="Create Transition",
                  command=create_transition).pack()

    def ask_for_rate(self):
        if len(self.arrows) < 1:
            return
        trans_name = simpledialog.askstring('Transition Name', 'Enter the name'
                                            ' of the transition for which you'
                                            ' want to add a rate function:')

        def create_params():
            choice = rate_function_var.get()
            if choice == "constant":
                params = 1
            elif (choice == "linear" or choice == "pexp" or choice == "nexp"):
                params = 2
            elif choice == "quadratic":
                params = 3
            else:
                params = None

            if params is not None:
                tk.messagebox.showinfo("Parameters", "For this transition will"
                                       f" be created {params} optimizable"
                                       " parameters")
            if not choice:
                return
            arrow = next((arrow for arrow in self.arrows if arrow.name ==
                          trans_name), None)
            if arrow:
                arrow.set_rate_function(choice)
            rate_window.destroy()

        rate_function_var = tk.StringVar()
        rate_function_var.set("Constant")
        rate_window = tk.Toplevel(self.master)
        rate_window.title("Add Rate Function")
        tk.Label(rate_window,
                 text="Which rate function for the transition?").pack()
        # Below a list of possible functions, must have 1 to 1 correspondence
        # with functions in Functions/functions.py
        tk.Radiobutton(rate_window, text="Constant: r0",
                       variable=rate_function_var, value="constant").pack()
        tk.Radiobutton(rate_window, text="Linear: r0+r1*x",
                       variable=rate_function_var, value="linear").pack()
        tk.Radiobutton(rate_window, text="Quadratic: r0+r1*x+r2*x^2",
                       variable=rate_function_var, value="quadratic").pack()
        tk.Radiobutton(rate_window, text="Pos exponential: r0*exp(r1*V*q/RT)",
                       variable=rate_function_var, value="pexp").pack()
        tk.Radiobutton(rate_window, text="Neg exponential: r0*exp(-r1*V*q/RT)",
                       variable=rate_function_var, value="nexp").pack()
        tk.Button(rate_window, text="Create Transition",
                  command=create_params).pack()

    def clear(self):
        for circle in self.circles.values():
            self.canvas.delete(circle.item)
            self.canvas.delete(circle.label)
        for arrow in self.arrows:
            self.canvas.delete(arrow.line)
            self.canvas.delete(arrow.label)
        self.circles = {}
        self.arrows = []

    def export_model(self):
        # Check if the Markov Model object is not empty
        if not self.circles or not self.arrows:
            tk.messagebox.showerror("Error", "The Markov Model is empty."
                                    " Add states and transitions before"
                                    " exporting.")
            return

        # Create a new MarkovModel object
        markov_model = MarkovModel()

        # Add states to the MarkovModel object
        for state in self.circles.values():
            markov_model.add_state(state.name)

        # Add transitions and rate functions to the MarkovModel object
        for arrow in self.arrows:
            from_state = arrow.start.name
            to_state = arrow.end.name
            rate_function = arrow.rate_function
            tname = arrow.name
            markov_model.add_transition(tname, from_state,
                                        to_state, rate_function)

        # Save the MarkovModel object to a pkl file
        file_path = asksaveasfilename(title="Save Markov Model",
                                      filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, "wb") as file:
                pkl.dump(markov_model, file)

        tk.messagebox.showinfo("Success", "Markov Model has been exported "
                               "to a pkl file.")

    def import_model(self):
        file_path = askopenfilename(title="Import Markov Model",
                                    filetypes=[("Pickle Files", "*.pkl")])

        if file_path:
            with open(file_path, "rb") as file:
                markov_model = pkl.load(file)

            # Clear the current canvas
            self.clear()
            tk.messagebox.showinfo("Success", "Markov Model imported ")
            # Create states from the MarkovModel object
            for state_name in markov_model.states:
                x = rand()*self.canvas.winfo_width()
                y = rand()*self.canvas.winfo_height()
                state = State(self.canvas, x, y, state_name)
                self.circles[state_name] = state

            # Create transitions from the MarkovModel object
            for tname, from_state, to_state, rate_function in markov_model.transitions:
                from_state_name = list(markov_model.states.keys())[from_state]
                to_state_name = list(markov_model.states.keys())[to_state]
                transition = Transition(self.canvas,
                                        self.circles[from_state_name],
                                        self.circles[to_state_name], tname)
                transition.set_rate_function(rate_function)
                self.arrows.append(transition)

    def show_table(self):
        # Clear the previous table
        for item in self.table_tree.get_children():
            self.table_tree.delete(item)

        # Add columns to the table
        self.table_tree['columns'] = ('Type', 'Name', 'From State', 'To State',
                                      'Rate Function')
        self.table_tree.column('#0', width=0, stretch=tk.NO)
        self.table_tree.column('Type', anchor=tk.W, width=100)
        self.table_tree.column('Name', anchor=tk.W, width=100)
        self.table_tree.column('From State', anchor=tk.W, width=100)
        self.table_tree.column('To State', anchor=tk.W, width=100)
        self.table_tree.column('Rate Function', anchor=tk.W, width=200)

        # Add headers
        self.table_tree.heading('#0', text='', anchor=tk.W)
        self.table_tree.heading('Type', text='Type', anchor=tk.W)
        self.table_tree.heading('Name', text='Name', anchor=tk.W)
        self.table_tree.heading('From State', text='From State', anchor=tk.W)
        self.table_tree.heading('To State', text='To State', anchor=tk.W)
        self.table_tree.heading('Rate Function', text='Rate Function',
                                anchor=tk.W)

        # Add states to the table, commented to avoid crowded table
#        for state in self.circles.values():
#            self.table_tree.insert('', tk.END, text='',
#                                   values=('State', state.name, '', ''))

        # Add transitions to the table
        for arrow in self.arrows:
            from_state = arrow.start.name
            to_state = arrow.end.name
            rate_function = arrow.rate_function
            tname = arrow.name
            self.table_tree.insert('', tk.END, text='',
                                   values=('Transition', tname, from_state,
                                           to_state, rate_function))

        # Update the table display
        self.table_tree.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = MMeditor(root)
    root.mainloop()
