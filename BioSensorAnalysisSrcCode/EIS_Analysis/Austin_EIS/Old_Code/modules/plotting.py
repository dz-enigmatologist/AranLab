import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_csv_data(df, sheet_name):
    """Plots CSV data using 'Rs' as x and absolute 'X' as y."""
    plt.figure()
    plt.plot(df["Rs"], np.abs(df["X"]), marker='o', linestyle='-')
    plt.xlabel("Rs")
    plt.ylabel("Abs(X)")
    plt.title(f"Plot for {sheet_name}")
    plt.grid(True)
    plt.show()

def plot_excel_sheet(ws):
    """
    Plots data from a worksheet using 'Rs' as x and absolute 'X' as y.
    Skips plotting if the required columns are not found.
    """
    data = list(ws.values)
    if not data or len(data) < 2:
        print(f"Worksheet {ws.title} has no data.")
        return

    header = data[0]
    df = pd.DataFrame(data[1:], columns=header)

    if "Rs" not in df.columns or "X" not in df.columns:
        print(f"Worksheet {ws.title}: Missing 'Rs' or 'X' columns.")
        return

    try:
        x = pd.to_numeric(df["Rs"], errors='coerce')
        y = pd.to_numeric(df["X"], errors='coerce').abs()

        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-')
        plt.title(f"Plot for {ws.title}")
        plt.xlabel("Rs")
        plt.ylabel("Abs(X)")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Plotting failed for {ws.title}: {e}")

