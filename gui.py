import tkinter as tk
from tkinter import filedialog
from Input.loader import plot_data

class PlotApp:

    def __init__(self, master):
        self.master = master
        master.title("Plot App")

        self.file_paths = []

        self.load_button = tk.Button(master, text="Load Files", command=self.load_files)
        self.load_button.pack()

        self.file_list = tk.Listbox(master)
        self.file_list.pack()

        self.plot_button = tk.Button(master, text="Plot", command=self.plot_data)
        self.plot_button.pack()

    def load_files(self):
        self.file_paths += filedialog.askopenfilenames()
        for file_path in filedialog.askopenfilenames():
            self.file_list.insert(tk.END, file_path)

    def plot_data(self):
        plot_data(self.file_paths)

root = tk.Tk()
app = PlotApp(root)
root.mainloop()

