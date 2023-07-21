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
        self.line = canvas.create_line(start.x, start.y, end.x, end.y,
                                       arrow='last')
        start.arrows.append(self)
        end.arrows.append(self)
        self.name = name
        self.label = canvas.create_text((start.x+end.x)/2, (start.y+end.y)/2,
                                        text=name)

    def redraw(self):
        self.canvas.coords(self.line, self.start.x, self.start.y, self.end.x,
                           self.end.y)
        self.canvas.coords(self.label, (self.start.x+self.end.x)/2,
                           (self.start.y+self.end.y)/2)


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Markov Model Editor")
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.circles = []
        self.arrows = []

        self.add_circle_button = tk.Button(master, text='Add State',
                                           command=self.add_circle)
        self.add_circle_button.pack(side=tk.LEFT)

        self.add_arrow_button = tk.Button(master, text='Add Transition',
                                          command=self.add_arrow)
        self.add_arrow_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(master, text='Clear All',
                                      command=self.clear)
        self.clear_button.pack(side=tk.LEFT)

    def add_circle(self):
        name = simpledialog.askstring('State Name', 'Enter a name for the'
                                      ' State:')
        if not name:
            return
        states = State(self.canvas, 250, 250, name)
        self.circles.append(states)

    def add_arrow(self):
        if len(self.circles) < 2:
            return
        start_index = simpledialog.askinteger('Start State', 'Enter the index'
                                              'of the starting state:')
        if start_index is None or start_index < 0 or \
           start_index >= len(self.circles):
            return
        end_index = simpledialog.askinteger('End State', 'Enter the index of'
                                            ' the ending state:')
        if end_index is None or end_index < 0 or \
           end_index >= len(self.circles) or end_index == start_index:
            return
        name = simpledialog.askstring('Transition Name', 'Enter name for '
                                      'the transition:')
        if not name:
            return
        start = self.circles[start_index]
        end = self.circles[end_index]
        arrow = Transition(self.canvas, start, end, name)
        self.arrows.append(arrow)

    def clear(self):
        for circle in self.circles:
            self.canvas.delete(circle.item)
            self.canvas.delete(circle.label)
        for arrow in self.arrows:
            self.canvas.delete(arrow.line)
            self.canvas.delete(arrow.label)
        self.circles = []
        self.arrows = []


root = tk.Tk()
app = App(root)
root.mainloop()
