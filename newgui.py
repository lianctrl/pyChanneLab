import tkinter as tk
from MM.MMgui import MMeditor  # Import the Markov Model Editor GUI module
from MM.MMgui import MarkovModel


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("pyChannel Lab")

        self.canvas = tk.Canvas(self, width=300, height=100)
        self.canvas.pack()

        # Create buttons or other widgets for the main GUI
        self.open_MMeditor_button = tk.Button(self,
                                              text="Open Markov Model Editor",
                                              command=self.open_MMeditor)
        self.open_MMeditor_button.pack()

    def open_MMeditor(self):
        # Create a Toplevel window for the Markov Model Editor
        editor_window = tk.Toplevel(self)
        editor_app = MMeditor(editor_window)


if __name__ == "__main__":
    main_app = MainApp()
    main_app.mainloop()
