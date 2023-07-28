import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import matplotlib as mpl


mpl.rcParams['figure.dpi']=180
mpl.rcParams['figure.titlesize']=20
mpl.rcParams['axes.facecolor']='white'
mpl.rcParams['lines.linewidth']=2.0
mpl.rcParams['axes.linewidth']=2.0
mpl.rcParams['xtick.major.pad']=8
mpl.rcParams['ytick.major.pad']=8
mpl.rcParams['ytick.minor.pad']=6
mpl.rcParams['xtick.labelsize']=10
mpl.rcParams['ytick.labelsize']=10
mpl.rcParams['axes.titlesize']=14
mpl.rcParams['axes.labelsize']=14
mpl.rc('text',usetex=False)
mpl.rcParams['axes.grid']='True'
mpl.rcParams['axes.axisbelow']='line'
mpl.rcParams['legend.loc']='best'
mpl.rcParams['legend.fontsize']=12


class HoldingPotential:
    def __init__(self, voltage, time):
        self.voltage = voltage
        self.time = time


class VariablePotential:
    def __init__(self, start_v, end_v, delta_v, time):
        self.start_v = start_v
        self.end_v = end_v
        self.delta_v = delta_v
        self.time = time

    def get_voltage_time_data(self):
        voltage_data = np.arange(self.start_v, self.end_v + self.delta_v,
                                 self.delta_v)
        time_data = np.full(len(voltage_data), self.time)
        return voltage_data, time_data


class VariableTime:
    def __init__(self, start_t, end_t, delta_t, value):
        self.start_t = start_t
        self.end_t = end_t
        self.delta_t = delta_t
        self.value = value

    def get_voltage_time_data(self):
        time_data = np.arange(self.start_t, self.end_t + self.delta_t,
                              self.delta_t)
        value_data = np.full(len(time_data), self.value)
        return value_data, time_data


class ExperimentBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Experiment Builder")

        self.holding_potentials = []
        self.times = None
        self.values = None

        self.voltage_label = tk.Label(root, text="Voltage (mV):")
        self.voltage_label.grid(row=0, column=0)
        self.voltage_entry = tk.Entry(root)
        self.voltage_entry.grid(row=0, column=1)

        self.time_label = tk.Label(root, text="Time (s):")
        self.time_label.grid(row=1, column=0)
        self.time_entry = tk.Entry(root)
        self.time_entry.grid(row=1, column=1)

        self.add_button = tk.Button(root, text="Add Holding Potential",
                                    command=self.add_holding_potential)
        self.add_button.grid(row=2, columnspan=2)

        self.holding_listbox = tk.Listbox(root)
        self.holding_listbox.grid(row=3, columnspan=2)
        self.holding_listbox.bind('<<ListboxSelect>>', self.on_select_holding)

        self.convert_button = tk.Button(root, text="Convert to Variable",
                                        command=self.open_variable_dialog)
        self.convert_button.grid(row=4, columnspan=2)

        self.clear_button = tk.Button(root, text="Clear Last Item",
                                      command=self.clear_last_item)
        self.clear_button.grid(row=5, columnspan=2)

        self.create_button = tk.Button(root, text="Create Experiment",
                                       command=self.create_experiment)
        self.create_button.grid(row=6, columnspan=2)

        self.export_button = tk.Button(root, text="Export Experiment",
                                       command=self.export_experiment)
        self.export_button.grid(row=7, columnspan=2)

        self.selected_holding_index = None

    def add_holding_potential(self):
        try:
            voltage = float(self.voltage_entry.get())
            time = float(self.time_entry.get())
            holding_potential = HoldingPotential(voltage, time)
            self.holding_potentials.append(holding_potential)
            self.holding_listbox.insert(tk.END, f"{voltage} mV, {time} s")
        except ValueError:
            pass

    def on_select_holding(self, event):
        selected_index = self.holding_listbox.curselection()
        if selected_index:
            self.selected_holding_index = selected_index[0]

    def open_variable_dialog(self):
        if self.selected_holding_index is not None:
            holding = self.holding_potentials[self.selected_holding_index]

            # Check if a variable protocol already exists
            if any(isinstance(h, VariablePotential) or isinstance(h, VariableTime) for h in self.holding_potentials):
                messagebox.showerror("Error", "Only one variable protocol can be inserted.")
                return

            # Check if the first item is a variable protocol
            if self.selected_holding_index == 0:
                messagebox.showerror("Error", "The first item cannot be a variable protocol.")
                return

            dialog = VariableInputDialog(self.root)
            self.root.wait_window(dialog.top)
            if dialog.result:
                variable_type = dialog.variable_type.get()
                if variable_type == "Potential":
                    start_v, end_v, delta_v = dialog.result
                    variable_potential = VariablePotential(start_v, end_v,
                                                           delta_v,
                                                           holding.time)
                    self.holding_potentials[self.selected_holding_index] = variable_potential
                    self.holding_listbox.delete(self.selected_holding_index)
                    self.holding_listbox.insert(self.selected_holding_index,
                                                f"Variable Potential, {holding.time} s")
                elif variable_type == "Time":
                    start_t, end_t, delta_t = dialog.result
                    variable_time = VariableTime(start_t, end_t, delta_t,
                                                 holding.voltage)
                    self.holding_potentials[self.selected_holding_index] = variable_time
                    self.holding_listbox.delete(self.selected_holding_index)
                    self.holding_listbox.insert(self.selected_holding_index,
                                                f"Variable Time, {holding.voltage} mV")

    def clear_last_item(self):
        if self.holding_potentials:
            self.holding_listbox.delete(tk.END)
            self.holding_potentials.pop()

    def create_experiment(self):
        for segment in self.holding_potentials:
            if isinstance(segment, VariablePotential):
                nfunc = len(segment.get_voltage_time_data()[0])
                break
            elif isinstance(segment, VariableTime):
                nfunc = len(segment.get_voltage_time_data()[0])
                break
            else:
                nfunc = 1

        voltages = np.zeros((nfunc, len(self.holding_potentials)+1))
        times = np.zeros((nfunc, len(self.holding_potentials)+1))

        for i in range(nfunc):
            for j, segment in enumerate(self.holding_potentials):
                if isinstance(segment, HoldingPotential):
                    if j == 0:
                        voltages[i][j] = segment.voltage

                    voltages[i][j+1] = segment.voltage
                    times[i][j+1] = times[i][j] + segment.time

                elif isinstance(segment, VariablePotential):
                    voltage_data, time_data = segment.get_voltage_time_data()
                    voltages[i][j+1] = voltage_data[i]
                    times[i][j+1] = times[i][j] + time_data[i]

                elif isinstance(segment, VariableTime):
                    voltage_data, time_data = segment.get_voltage_time_data()
                    voltages[i][j+1] = voltage_data[i]
                    times[i][j+1] = times[i][j] + time_data[i]

        self.times = times
        self.voltages = voltages

        for i in range(nfunc):
            plt.step(times[i, :], voltages[i, :])

        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.grid(True)
        plt.show()

    def export_experiment(self):
        if self.times is None or self.voltages is None:
            messagebox.showinfo("Error", "No data to export.")
            return

        # Get the file path from the user
        file_path = asksaveasfilename(title="Export Experiment",
                                      filetypes=[("Pickle Files",
                                                  "*.pkl")])
        if file_path:
            # Create a dictionary to save times and voltages arrays
            data_dict = {"times": self.times, "voltages": self.voltages}

            # Save the data to a pickle file
            with open(file_path, "wb") as file:
                pkl.dump(data_dict, file)

            messagebox.showinfo("Success", "Experiment protocol has been"
                                           "exported to file.")


class VariableInputDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        self.top.title("Variable Input")

        self.variable_type = tk.StringVar()
        self.variable_type.set("Potential")

        self.type_label = tk.Label(top, text="Choose Variable Type:")
        self.type_label.grid(row=0, column=0)
        self.type_option_menu = tk.OptionMenu(top, self.variable_type,
                                              "Potential", "Time")
        self.type_option_menu.grid(row=0, column=1)

        self.start_label = tk.Label(top, text="Start Value:")
        self.start_label.grid(row=1, column=0)
        self.start_entry = tk.Entry(top)
        self.start_entry.grid(row=1, column=1)

        self.end_label = tk.Label(top, text="End Value:")
        self.end_label.grid(row=2, column=0)
        self.end_entry = tk.Entry(top)
        self.end_entry.grid(row=2, column=1)

        self.delta_label = tk.Label(top, text="Delta Value:")
        self.delta_label.grid(row=3, column=0)
        self.delta_entry = tk.Entry(top)
        self.delta_entry.grid(row=3, column=1)

        self.ok_button = tk.Button(top, text="OK", command=self.ok)
        self.ok_button.grid(row=4, columnspan=2)

        self.result = None

    def ok(self):
        variable_type = self.variable_type.get()
        try:
            if variable_type == "Potential":
                start_v = float(self.start_entry.get())
                end_v = float(self.end_entry.get())
                delta_v = float(self.delta_entry.get())
                self.result = (start_v, end_v, delta_v)
            elif variable_type == "Time":
                start_t = float(self.start_entry.get())
                end_t = float(self.end_entry.get())
                delta_t = float(self.delta_entry.get())
                self.result = (start_t, end_t, delta_t)
            self.top.destroy()
        except ValueError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentBuilderApp(root)
    root.mainloop()
