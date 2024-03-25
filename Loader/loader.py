import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
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
        self.load_button.pack(pady=10)

        # Plot Button
        self.plot_button = tk.Button(parent,
                                     text="Plot Data",
                                     command=self.plot_data, 
                                     state='disabled')
        self.plot_button.pack(pady=10)

        # Export Button
        self.export_button = tk.Button(parent,
                                     text="Export Data to be fit",
                                     command=self.export_data, 
                                     state='disabled')
        self.export_button.pack(pady=10)

        # Compare Button
        self.compare_button = tk.Button(parent,
                                       text="Compare Experiments and fitting",
                                       command=self.compare_data) 
        self.compare_button.pack(pady=10)

        self.curves_list = []
        self.data = None

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not file_path:
            return


        self.curves_list.append(file_path)  # Add valid file paths to the list
        
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
            self.export_button.config(state='normal')

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

    def export_data(self):
        data_arrays = []
        for file_path in self.curves_list:
            try:
                data = pd.read_csv(file_path, header=0).to_numpy()
                if data.shape[1] == 2 or data.shape[1] == 3:
                    data_arrays.append(data[:, :2])  # Ensure only x, y columns are considered
                else:
                    messagebox.showwarning("Warning", f"File {file_path} does not have exactly 2 columns (x, y). Skipping.")
                    continue
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file {file_path}: {e}")
                continue

        if data_arrays:
            export_file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
            if not export_file_path:
                return
            with open(export_file_path, 'wb') as file:
                pickle.dump(data_arrays, file)
            messagebox.showinfo("Export Successful", f"Curves list has been exported to {export_file_path}")
        else:
            messagebox.showerror("Export Failed", "No valid data to export.")

    def compare_data(self):
        # Ask the user to select the experimental data file
        exp_file_path = filedialog.askopenfilename(title="Select Experimental Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not exp_file_path:
            messagebox.showinfo("Info", "Experimental data file selection cancelled.")
            return

        # Ask the user to select the fitted data file
        fitted_file_path = filedialog.askopenfilename(title="Select Fitted Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not fitted_file_path:
            messagebox.showinfo("Info", "Fitted data file selection cancelled.")
            return

        try:
            # Load the experimental data
            exp_data = pd.read_csv(exp_file_path, header=0)
            self.x_label = exp_data.columns[0]  # Name of the first column for x-axis label
            self.y_label = exp_data.columns[1]  # Name of the second column for y-axis label
            exp_x, exp_y = exp_data.iloc[:, 0], exp_data.iloc[:, 1]
            if exp_data.shape[1] == 3:
                err_y = exp_data.iloc[:, 2]

            # Load the fitted data

            fitted_data = pd.read_csv(fitted_file_path, header=0)
            fitted_x, fitted_y = fitted_data.iloc[:, 0], fitted_data.iloc[:, 1]

            # Plotting
            plt.figure(figsize=(10, 6))
            if exp_data.shape[1] == 2:
                plt.scatter(exp_x, exp_y, label='Experimental', marker='o', color='blue')
            elif exp_data.shape[1] == 3:
                plt.errorbar(exp_x, exp_y, err_y, label='Experimental', fmt='o', color='blue')
            plt.scatter(fitted_x, fitted_y, label='Fitted', marker='s', color='red')
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load and compare data: {e}")




if __name__ == "__main__":
    root = tk.Tk()
    app = DataLoader(root)
    root.mainloop()