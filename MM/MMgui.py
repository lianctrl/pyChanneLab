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
    def __init__(self, canvas, start, end, name, two_way=False):
        self.canvas = canvas
        self.start = start
        self.end = end
        self.two_way = two_way
        self.line = canvas.create_line(start.x, start.y, end.x, end.y,
                                       arrow='last')
        start.arrows.append(self)
        end.arrows.append(self)
        self.name = name
        self.label_up = canvas.create_text(0, 0, text=name)
        self.label_down = canvas.create_text(0, 0, text=name)
        self.redraw()

    def redraw(self):
        self.canvas.coords(self.line, self.start.x, self.start.y, self.end.x, self.end.y)
        mid_x = (self.start.x + self.end.x) / 2
        mid_y = (self.start.y + self.end.y) / 2

        angle = tk.Canvas(self.canvas).create_line(self.start.x, self.start.y, self.end.x, self.end.y)
        angle = tk.Canvas(self.canvas).itemcget(angle, 'angle')
        angle = float(angle)

        label_distance = 15

        angle_offset = 30
        self.canvas.coords(self.label_up, mid_x + label_distance * math.cos(math.radians(angle + angle_offset)),
                           mid_y + label_distance * math.sin(math.radians(angle + angle_offset)))
        self.canvas.coords(self.label_down, mid_x + label_distance * math.cos(math.radians(angle - angle_offset)),
                           mid_y + label_distance * math.sin(math.radians(angle - angle_offset)))

    def delete(self):
        self.canvas.delete(self.line)
        self.canvas.delete(self.label_up)
        self.canvas.delete(self.label_down)


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
                                          command=self.add_arrow)
        self.add_arrow_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(master, text='Clear All',
                                      command=self.clear)
        self.clear_button.pack(side=tk.LEFT)

    def add_circle(self):
        name = simpledialog.askstring('State Name', 'Enter a name for the State:')
        if not name:
            return
        x, y = 250, 250
        states = State(self.canvas, x, y, name)
        self.circles[name] = states

    def add_arrow(self):
        if len(self.circles) < 2:
            return
        start_name = simpledialog.askstring('Start State', 'Enter the name of the starting state:')
        if not start_name or start_name not in self.circles:
            return
        end_name = simpledialog.askstring('End State', 'Enter the name of the ending state:')
        if not end_name or end_name not in self.circles or end_name == start_name:
            return
        name = simpledialog.askstring('Transition Name', 'Enter name for the transition:')
        if not name:
            return

        def two_way_yes():
            self.create_arrow(start_name, end_name, name, two_way=True)

        def two_way_no():
            self.create_arrow(start_name, end_name, name, two_way=False)

        two_way_frame = tk.Toplevel()
        two_way_frame.title("Two-Way Transition")
        two_way_label = tk.Label(two_way_frame, text="Create a two-way transition?")
        two_way_label.pack()
        two_way_yes_button = tk.Button(two_way_frame, text="Yes", command=two_way_yes)
        two_way_yes_button.pack(side=tk.LEFT)
        two_way_no_button = tk.Button(two_way_frame, text="No", command=two_way_no)
        two_way_no_button.pack(side=tk.LEFT)

    def create_arrow(self, start_name, end_name, name, two_way):
        start = self.circles[start_name]
        end = self.circles[end_name]
        arrow = Transition(self.canvas, start, end, name, two_way=two_way)
        self.arrows.append(arrow)
        if two_way:
            arrow2 = Transition(self.canvas, end, start, name, two_way=False)
            self.arrows.append(arrow2)

    def clear(self):
        for circle in self.circles.values():
            self.canvas.delete(circle.item)
            self.canvas.delete(circle.label)
        for arrow in self.arrows:
            arrow.delete()
        self.circles = {}
        self.arrows = []


root = tk.Tk()
app = App(root)
root.mainloop()
