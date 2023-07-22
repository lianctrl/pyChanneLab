import tkinter as tk
import tkinter.simpledialog as simpledialog


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
        self.canvas.coords(self.label, (self.start.x+self.end.x)/2,
                           (self.start.y+self.end.y)/2)

    def set_rate_function(self, rate_function):
        self.rate_function = rate_function
        self.name = self.name + f" ({rate_function})"
        self.canvas.itemconfig(self.label, text=self.name, fill='black')


class App:
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

    def add_circle(self):
        name = simpledialog.askstring('State Name', 'Enter a name'
                                      ' for the State:')
        if not name:
            return
        x, y = 250, 250
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
        if not end_name or end_name not in self.circles or end_name == start_name:
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
                                           ' back')
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


root = tk.Tk()
app = App(root)
root.mainloop()
