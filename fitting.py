import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import pickle as pkl

class MSMFittingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("MSM Fitting GUI")

        # Load MSM Button
        self.load_msm_button = tk.Button(master, text="Load MSM", command=self.load_msm)
        self.load_msm_button.pack()

        # Load Experiment Button
        self.load_experiment_button = tk.Button(master, text="Load Experiment", command=self.load_experiment)
        self.load_experiment_button.pack()

        # Load Experimental Curves Button
        self.load_curves_button = tk.Button(master, text="Load Curves", command=self.load_curves)
        self.load_curves_button.pack()

        # Optimization Parameters Button
        self.set_parameters_button = tk.Button(master, text="Set Optimization Parameters", command=self.set_parameters)
        self.set_parameters_button.pack()

        # Start Fitting Button
        self.start_fitting_button = tk.Button(master, text="Start Fitting", command=self.start_fitting)
        self.start_fitting_button.pack()

        # Attributes to store loaded data
        self.msm = None
        self.experiment = None
        self.curves = None
        self.optimization_parameters = None

    def load_msm(self):
        file_path = filedialog.askopenfilename(title="Select MSM file", filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, "rb") as file:
                self.msm = pkl.load(file)
            print("MSM loaded successfully.")

    def load_experiment(self):
        file_path = filedialog.askopenfilename(title="Select Experiment file", filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, "rb") as file:
                self.experiment = pkl.load(file)
            print("Experiment loaded successfully.")

    def load_curves(self):
        file_path = filedialog.askopenfilename(title="Select Curves file", filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, "rb") as file:
                self.curves = pkl.load(file)
            print("Curves loaded successfully.")

    def set_parameters(self):
        # This is a simplified example; you may need a dialog to capture multiple parameters
        max_iter = simpledialog.askinteger("Input", "Enter max number of iterations", minvalue=1)
        if max_iter is not None:
            self.optimization_parameters = {'maxiter': max_iter}
            print(f"Optimization parameters set: {self.optimization_parameters}")

    def start_fitting(self):
        if not self.msm or not self.experiment or not self.curves:
            print("Error: Please load all required files.")
            return
        if not self.optimization_parameters:
            print("Error: Please set optimization parameters.")
            return

        # Your fitting logic goes here, use self.msm, self.experiment, self.curves, and self.optimization_parameters
        print("Starting fitting process...")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = MSMFittingGUI(root)
    root.mainloop()
