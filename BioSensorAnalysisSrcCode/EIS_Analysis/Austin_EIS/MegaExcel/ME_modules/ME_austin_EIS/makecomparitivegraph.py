import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your lists
workbook_names = [ 'Chip 21', 'Chip 22', 'Chip 23', 'Chip 26', 'Chip 27', 'Chip 32', 'Chip 33', 'Chip 35', 'Chip 36', 'Chip 37', 'Chip 39', 'Chip 40', 'Chip 41', 'Chip 43', 'Chip 44', 'Chip 45', 'Chip 46', 'Chip 47', 'Chip 48', 'Chip 52', 'Chip 53', 'Chip 54', 'Chip 55', 'Chip 56', 'Chip 57', 'Chip 59' ]
worksheet_names = ["0pM_asso", "0pM_disso", "100pM_asso", "100pM_disso", "1nM_asso", "1nM_disso", "10nM_asso", "10nM_disso", "100nM_asso", "100nM_disso"]
headers = [ 'time(s)', 'delta Rct-a', 'Cp1', 'Ph1','linear_eq_slope', 'linear_eq_b', 'Rs', 'delta Rct-i', 'Q', 'n']

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
            plt.boxplot(data_for_plot, tick_labels=chip_labels)
            plt.tight_layout()

            # Save in correct subfolder
            safe_name = f"{ws}_{h}".replace('/', '-').replace('\\', '-')
            save_path = os.path.join(worksheet_folders[ws], f"{safe_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot: {save_path}")

def plot_vs_time(data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Subfolder for time-based plots
    time_plots_folder = os.path.join(output_folder, "time_vs_column_graphs")
    os.makedirs(time_plots_folder, exist_ok=True)

    for ws, header_data in data.items():
        if 'time(s)' not in header_data:
            continue

        time_values_per_chip = header_data['time(s)']

        for h in headers:
            if h == 'time(s)':
                continue

            plt.figure(figsize=(10, 6))
            plt.title(f"{ws} - {h} vs. time(s) (normalized)")
            plt.xlabel("time(s)")
            plt.ylabel(f"Normalized {h}")
            plt.grid(True, linestyle="--", alpha=0.5)

            any_data = False

            for chip_name, time_vals, y_vals in zip(workbook_names, time_values_per_chip, header_data[h]):
                if len(time_vals) == 0 or len(y_vals) == 0:
                    continue

                try:
                    time_numeric = np.array(time_vals, dtype=float)
                    y_numeric = np.array(y_vals, dtype=float)
                except:
                    continue

                if len(time_numeric) != len(y_numeric):
                    continue
                
                # Normalize y_numeric to range [0, 1]
                y_min = np.min(y_numeric)
                y_max = np.max(y_numeric)
                if y_max > y_min:
                    y_normalized = (y_numeric - y_min) / (y_max - y_min)
                else:
                    # If all values are the same, just use zeros
                    y_normalized = np.zeros_like(y_numeric)

                plt.plot(time_numeric, y_normalized, label=chip_name)
                any_data = True

            if not any_data:
                plt.close()
                continue

            plt.legend(fontsize=8)
            plt.tight_layout()

            safe_name = f"{ws}_{h}_vs_time_normalized".replace('/', '-').replace('\\', '-')
            save_path = os.path.join(time_plots_folder, f"{safe_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved normalized time-based plot: {save_path}")




def comparitive_graphs():
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    output_folder = os.path.abspath("Processed Chips for PCAI1ia")
    data = read_workbook_data(folder_path)
    plot_and_save(data, output_folder)
    plot_vs_time(data, output_folder)  # Add this line
    print(f"Plots saved in {output_folder}")


comparitive_graphs()
