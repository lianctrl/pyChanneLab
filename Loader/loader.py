import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataLoader:
    def __init__(self, parent):
        self.parent = parent 
        self.parent.title("Load Experimental Curves")
        self.parent.geometry("800x400")

        # Load Button
        self.load_button = tk.Button(parent, 
                                     text="Load Data", 
                                     command=self.load_data)
        self.load_button.pack(pady=20)

        # Plot Button
        self.plot_button = tk.Button(parent,
                                     text="Plot Data",
                                     command=self.plot_data, 
                                     state='disabled')
        self.plot_button.pack(pady=20)

        self.data = None

#    def load_data(self):
#        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
#        if not file_path:
#            return
#
#        try:
#            # Assuming the file is in CSV format
#            self.data = pd.read_csv(file_path)
#            if self.data.shape[1] < 2 or self.data.shape[1] > 3:
#                messagebox.showerror("Error", "File must have 2 or 3 columns (x, y, [error])")
#                self.data = None
#                return
#
#            messagebox.showinfo("Success", "Data loaded successfully")
#            self.plot_button.config(state='normal')
#        except Exception as e:
#            messagebox.showerror("Error", f"Failed to load file: {e}")
#            self.data = None
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not file_path:
            return

        try:
            # Load the data assuming the first row is a header with column names
            self.data = pd.read_csv(file_path, header=0)

            # Validate the data to ensure it has 2 or 3 columns
            if self.data.shape[1] < 2 or self.data.shape[1] > 3:
                messagebox.showerror("Error", "File must have 2 or 3 columns (x, y, [error])")
                self.data = None
                return

            # Check if data columns can be converted to numeric, if not raise an error
            try:
                self.data = self.data.apply(pd.to_numeric, errors='raise')
            except ValueError as e:
                messagebox.showerror("Error", f"Non-numeric data encountered: {e}")
                self.data = None
                return

            self.x_label = self.data.columns[0]  # Name of the first column for x-axis label
            self.y_label = self.data.columns[1]  # Name of the second column for y-axis label

            messagebox.showinfo("Success", "Data loaded successfully. Axis labels extracted.")
            self.plot_button.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.data = None

    def plot_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        fig, ax = plt.subplots()
        ax.plot(self.data.iloc[:, 0], self.data.iloc[:, 1], label="Experimental Curve")

        if self.data.shape[1] == 3:
            ax.errorbar(self.data.iloc[:, 0], self.data.iloc[:, 1], yerr=self.data.iloc[:, 2], fmt='o', label="Error")

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataLoader(root)
    root.mainloop()