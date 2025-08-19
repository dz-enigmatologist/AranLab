import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your lists
workbook_names = [ 'Chip 21', 'Chip 22', 'Chip 23', 'Chip 26', 'Chip 27', 'Chip 32', 'Chip 33', 'Chip 35', 'Chip 36', 'Chip 37', 'Chip 39', 'Chip 40', 'Chip 41', 'Chip 43', 'Chip 44', 'Chip 45', 'Chip 46', 'Chip 47', 'Chip 48', 'Chip 52', 'Chip 53', 'Chip 54', 'Chip 55', 'Chip 56', 'Chip 57', 'Chip 59' ]
worksheet_names = ["0pM_asso", "0pM_disso", "100pM_asso", "100pM_disso", "1nM_asso", "1nM_disso", "10nM_asso", "10nM_disso", "100nM_asso", "100nM_disso"]
headers = [ 'time(s)', 'delta Rct-a', 'Rct-d', 'Cp1', 'Ph1', 'Slope 1', 'Slope 2', 'Slope 3', 'Slope 4', 'Slope 5', 'Angle', 'Cp_exp-a', 'Cp_exp-b', 'Ph_slope', 'Ph_peak', 'Area Cp', 'Area Ph', 'Area Slope', 'Area Rs-direct', 'Area Rs-Para', '','','linear_eq_slope', 'linear_eq_b', 'Rs', 'delta Rct-i', 'Q', 'n']

def select_folder():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select folder containing workbooks")

def read_workbook_data(folder_path):
    data = {}  # {worksheet: {header: [values across chips]}}

    for chip_name in workbook_names:
        file_path = os.path.join(folder_path, f"{chip_name}.xlsx")
        if not os.path.exists(file_path):
            print(f"Workbook not found: {file_path}")
            continue

        wb = pd.ExcelFile(file_path)
        for ws in worksheet_names:
            if ws not in wb.sheet_names:
                continue
            df = wb.parse(ws)
            if ws not in data:
                data[ws] = {h: [] for h in headers}
            for h in headers:
                if h in df.columns:
                    data[ws][h].append(df[h].dropna().values)
                else:
                    data[ws][h].append([])

    return data


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_and_save(data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Subfolder names based on worksheet_names
    worksheet_folders = {ws: os.path.join(output_folder, f"{ws}_graphs") for ws in worksheet_names}
    for folder in worksheet_folders.values():
        os.makedirs(folder, exist_ok=True)

    for ws, header_data in data.items():
        for h, all_values in header_data.items():
            # Prepare DataFrame
            combined = []
            for chip_name, vals in zip(workbook_names, all_values):
                # Filter out empty or non-numeric values
                numeric_vals = []
                for v in vals:
                    try:
                        numeric_vals.append(float(v))
                    except (ValueError, TypeError):
                        continue
                if len(numeric_vals) > 0:
                    combined.append(pd.DataFrame({'Chip': chip_name, 'Value': numeric_vals}))

            if not combined:
                continue

            df = pd.concat(combined, ignore_index=True)

            if df['Value'].dropna().empty:
                continue

            # Prepare data for boxplot
            data_for_plot = []
            chip_labels = []
            for chip in df['Chip'].unique():
                values = df[df['Chip'] == chip]['Value'].dropna().astype(float)
                if len(values) > 0:
                    data_for_plot.append(values)
                    chip_labels.append(chip)

            if len(data_for_plot) < 2:
                continue

            # Plot
            plt.figure(figsize=(10, 6))
            plt.title(f"{ws} - {h}")
            plt.ylabel(h)
            plt.xlabel("Chip")
            plt.xticks(rotation=90)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.boxplot(data_for_plot, labels=chip_labels)
            plt.tight_layout()

            # Save in correct subfolder
            safe_name = f"{ws}_{h}".replace('/', '-').replace('\\', '-')
            save_path = os.path.join(worksheet_folders[ws], f"{safe_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot: {save_path}")


def comparitive_graphs ():
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    output_folder = os.path.abspath("Processed Chips for PCAI1ia")
    data = read_workbook_data(folder_path)
    plot_and_save(data, output_folder)
    print(f"Plots saved in {output_folder}")

comparitive_graphs()