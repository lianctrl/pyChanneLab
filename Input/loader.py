import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(file_paths):
    num_files = len(file_paths)
    fig, axes = plt.subplots(nrows=num_files, ncols=2, figsize=(8, num_files*4)), sharex=True)

    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path, header=1)
        if len(df.columns) == 2:
            x_vals = df.iloc[:, 0]
            y_vals = df.iloc[:, 1]
            y_errs = None
        elif len(df.columns) == 3:
            x_vals = df.iloc[:, 0]
            y_vals = df.iloc[:, 1]
            y_errs = df.iloc[:, 2]
        else:
            raise ValueError(f"File {file_path} has {len(df.columns)} columns, expected 2 or 3.")

        ax = axes[i]
        if y_errs is None:
            ax.plot(x_vals, y_vals, 'o')
        else:
            ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o')
        ax.set_title(file_path)

    plt.tight_layout()
    plt.show()
