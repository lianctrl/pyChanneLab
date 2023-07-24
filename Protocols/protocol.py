
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


class Protocol:
    def __init__(self, name, type, value, delta, time, delta_time):
        self.name = name
        self.type = type
        self.value = value
        self.delta = delta
        self.time = time
        self.delta_time = delta_time


class ProtocolCreatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Protocol Creator")

        self.name_var = tk.StringVar()
        self.type_var = tk.StringVar()
        self.value_var = tk.DoubleVar()
        self.delta_var = tk.DoubleVar()
        self.time_var = tk.DoubleVar()
        self.delta_time_var = tk.DoubleVar()

        self.name_label = tk.Label(master, text="Name:")
        self.name_label.pack()
        self.name_entry = tk.Entry(master, textvariable=self.name_var)
        self.name_entry.pack()

        self.type_label = tk.Label(master, text="Type:")
        self.type_label.pack()
        self.type_combobox = ttk.Combobox(master, textvariable=self.type_var,
                                          values=["Voltage", "Concentration"])
        self.type_combobox.pack()

        self.value_label = tk.Label(master, text="Value:")
        self.value_label.pack()
        self.value_entry = tk.Entry(master, textvariable=self.value_var)
        self.value_entry.pack()

        self.delta_label = tk.Label(master, text="Delta:")
        self.delta_label.pack()
        self.delta_entry = tk.Entry(master, textvariable=self.delta_var)
        self.delta_entry.pack()

        self.time_label = tk.Label(master, text="Time:")
        self.time_label.pack()
        self.time_entry = tk.Entry(master, textvariable=self.time_var)
        self.time_entry.pack()

        self.delta_time_label = tk.Label(master, text="Delta Time:")
        self.delta_time_label.pack()
        self.delta_time_entry = tk.Entry(master,
                                         textvariable=self.delta_time_var)
        self.delta_time_entry.pack()

        self.create_button = tk.Button(master, text="Create Protocol",
                                       command=self.create_protocol)
        self.create_button.pack()

        self.protocol_details_label = tk.Label(master, text="")
        self.protocol_details_label.pack()

    def create_protocol(self):
        name = self.name_var.get()
        type_ = self.type_var.get()
        value = self.value_var.get()
        delta = self.delta_var.get()
        time = self.time_var.get()
        delta_time = self.delta_time_var.get()

        if not name or not type_:
            messagebox.showerror("Error", "Name and Type cannot be empty.")
            return

        protocol = Protocol(name, type_, value, delta, time, delta_time)
        self.show_protocol_details(protocol)

    def show_protocol_details(self, protocol):
        details = f"Protocol: {protocol.name}\n" \
                  f"Type: {protocol.type}\n" \
                  f"Value: {protocol.value}\n" \
                  f"Delta: {protocol.delta}\n" \
                  f"Time: {protocol.time}\n" \
                  f"Delta Time: {protocol.delta_time}"
        self.protocol_details_label.config(text=details)


if __name__ == "__main__":
    root = tk.Tk()
    app = ProtocolCreatorApp(root)
    root.mainloop()
