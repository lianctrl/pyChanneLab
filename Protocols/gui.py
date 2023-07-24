import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt


class SubProtocol:
    def __init__(self, sub_type, values, time):
        self.sub_type = sub_type
        self.values = values
        self.time = time


class VoltageProtocol:
    def __init__(self, name):
        self.name = name
        self.sub_protocols = []

    def add_sub_protocol(self, sub_protocol):
        self.sub_protocols.append(sub_protocol)


class ProtocolCreatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Protocol Creator")

        self.protocols = []  # List to store the voltage protocols

        self.name_var = tk.StringVar()

        self.name_label = tk.Label(master, text="Protocol Name:")
        self.name_label.pack()
        self.name_entry = tk.Entry(master, textvariable=self.name_var)
        self.name_entry.pack()

        self.add_holding_button = tk.Button(master, text="Add Holding Sub-Protocol",
                                            command=self.add_holding_sub_protocol)
        self.add_holding_button.pack()

        self.add_variable_button = tk.Button(master, text="Add Variable Sub-Protocol",
                                             command=self.add_variable_sub_protocol)
        self.add_variable_button.pack()

        self.protocol_details_label = tk.Label(master, text="")
        self.protocol_details_label.pack()

        self.protocol_table = ttk.Treeview(master, columns=("Name", "Type"))
        self.protocol_table.heading("#0", text="Index")
        self.protocol_table.heading("Name", text="Name")
        self.protocol_table.heading("Type", text="Type")
        self.protocol_table.pack()

        self.create_experiment_button = tk.Button(master, text="Create Experiment", command=self.show_experiment_window)
        self.create_experiment_button.pack()

    def add_holding_sub_protocol(self):
        voltage = simpledialog.askfloat("Holding Sub-Protocol", "Enter Holding Voltage (or Concentration):")
        if voltage is None:
            return

        time = simpledialog.askfloat("Holding Sub-Protocol", "Enter Holding Time:")
        if time is None:
            return

        sub_protocol = SubProtocol(sub_type="Holding", values=[voltage], time=time)
        protocol = VoltageProtocol(name=self.name_var.get())
        protocol.add_sub_protocol(sub_protocol)
        self.protocols.append(protocol)
        self.update_protocol_table()

    def add_variable_sub_protocol(self):
        sub_type = simpledialog.askstring("Variable Sub-Protocol",
                                          "Enter Variable Type (Voltage/Concentration) or time:")
        if sub_type is None or sub_type.lower() not in ["voltage", "concentration", "time"]:
            return

        values_str = simpledialog.askstring("Variable Sub-Protocol",
                                            "Enter Variable Values (comma-separated):")
        if values_str is None:
            return
        if sub_type in ["voltage", "concentration"]:
            fix_val = simpledialog.askfloat("Variable Sub-Protocol", "Enter time of each sweep:")
        elif sub_type in ["time"]:
            fix_val = simpledialog.askfloat("Variable Sub-Protocol", "Enter voltage/concentration of each sweep:")
        else:
            return
        if fix_val is None:
            return

        values = [float(val.strip()) for val in values_str.split(",")]

        sub_protocol = SubProtocol(sub_type=sub_type.capitalize(),
                                   values=values, time=fix_val)
        protocol = VoltageProtocol(name=self.name_var.get())
        protocol.add_sub_protocol(sub_protocol)
        self.protocols.append(protocol)
        self.update_protocol_table()

    def show_protocol_details(self, protocol):
        details = f"Protocol: {protocol.name}\n"
        for idx, sub_protocol in enumerate(protocol.sub_protocols, start=1):
            details += f"Sub-Protocol {idx}: {sub_protocol.sub_type}\n" \
                       f"Values: {', '.join(map(str, sub_protocol.values))}\n" \
                       f"Time: {sub_protocol.time}\n\n"
        self.protocol_details_label.config(text=details)

    def update_protocol_table(self):
        # Clear previous data
        self.protocol_table.delete(*self.protocol_table.get_children())

        # Insert protocols into the table
        for idx, protocol in enumerate(self.protocols):
            self.protocol_table.insert("", "end", values=(idx + 1, protocol.name))

    def show_experiment_window(self):
        if not self.protocols:
            messagebox.showerror("Error", "No protocols to create an experiment.")
            return

        experiment_window = tk.Toplevel(self.master)
        experiment_window.title("Create Experiment")

        selected_protocols = []

        def add_protocol():
            selected_idx = protocol_listbox.curselection()
            for idx in selected_idx:
                selected_protocols.append(self.protocols[idx])

        def create_experiment():
            # Here you can implement the logic for creating an experiment
            # using the selected_protocols list.
            if selected_protocols:
                protocol_names = [protocol.name for protocol in selected_protocols]
                messagebox.showinfo("Experiment Created",
                                    f"Experiment created with protocols: {', '.join(protocol_names)}")

                # You can perform additional actions for the experiment, such as plotting.
                # Please specify your requirements, and I can assist you further.

                experiment_window.destroy()
            else:
                messagebox.showerror("Error", "No protocols selected for the experiment.")

        protocol_listbox = tk.Listbox(experiment_window, selectmode=tk.MULTIPLE)
        for protocol in self.protocols:
            protocol_listbox.insert(tk.END, protocol.name)
        protocol_listbox.pack()

        add_button = tk.Button(experiment_window, text="Add Protocol", command=add_protocol)
        add_button.pack()

        confirm_button = tk.Button(experiment_window, text="Confirm", command=create_experiment)
        confirm_button.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProtocolCreatorApp(root)
    root.mainloop()
