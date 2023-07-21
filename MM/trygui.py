import tkinter as tk
import tkinter.simpledialog as simpledialog


class Circle:
    def __init__(self, canvas, x, y, name):
        self.canvas = canvas
        self.item = canvas.create_oval(x-20, y-20, x+20, y+20, fill='white', outline='black')
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


class Arrow:
    def __init__(self, canvas, start, end, name):
        self.canvas = canvas
        self.start = start
        self.end = end
        self.line = canvas.create_line(start.x, start.y, end.x, end.y, arrow='last')
        start.arrows.append(self)
        end.arrows.append(self)
        self.name = name
        self.label = canvas.create_text((start.x+end.x)/2, (start.y+end.y)/2, text=name)

    def redraw(self):
        self.canvas.coords(self.line, self.start.x, self.start.y, self.end.x, self.end.y)
        self.canvas.coords(self.label, (self.start.x+self.end.x)/2, (self.start.y+self.end.y)/2)


class App:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.circles = []
        self.arrows = []

        self.add_circle_button = tk.Button(master, text='Add Circle', command=self.add_circle)
        self.add_circle_button.pack(side=tk.LEFT)

        self.add_arrow_button = tk.Button(master, text='Add Arrow', command=self.add_arrow)
        self.add_arrow_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(master, text='Clear All', command=self.clear)
        self.clear_button.pack(side=tk.LEFT)

    def add_circle(self):
        name = simpledialog.askstring('Circle Name', 'Enter a name for the circle:')
        if not name:
            return
        circle = Circle(self.canvas, 250, 250, name)
        self.circles.append(circle)

    def add_arrow(self):
        if len(self.circles) < 2:
            return
        start_index = simpledialog.askinteger('Start Circle', 'Enter the index of the start circle:')
        if start_index is None or start_index < 0 or start_index >= len(self.circles):
            return
        end_index = simpledialog.askinteger('End Circle', 'Enter the index of the end circle:')
        if end_index is None or end_index < 0 or end_index >= len(self.circles) or end_index == start_index:
            return
        name = simpledialog.askstring('Arrow Name', 'Enter a name for the arrow:')
        if not name:
            return
        start = self.circles[start_index]
        end = self.circles[end_index]
        arrow = Arrow(self.canvas, start, end, name)
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
