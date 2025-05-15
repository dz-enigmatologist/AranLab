import numpy as np
import pandas as pd
import Anal_CSV as ac

def split_cycles_from_frequency(frequency, start_val=100, end_val=200000, tol_start=20, tol_end=10000):
    """
    Splits the frequency array into cycles using relaxed tolerances.
    A cycle starts near `start_val` and ends near `end_val`.
    """
    print(f"[Cycle Split] Looking for cycles: start ~{start_val}±{tol_start}, end ~{end_val}±{tol_end}")
    print(f"[Cycle Split] First 10 frequencies: {frequency[:10]}")

    cycles = []
    in_cycle = False
    start_idx = None
    for i, freq in enumerate(frequency):
        if not in_cycle:
            if abs(freq - start_val) <= tol_start:
                in_cycle = True
                start_idx = i
        else:
            if abs(freq - end_val) <= tol_end:
                cycles.append((start_idx, i + 1))
                in_cycle = False
                start_idx = None
    return cycles

def process_excel_cycles(ws):
    """
    Processes an Excel worksheet that contains continuous cycle data.
    Looks for flexible 'Frequency' column and logs each cycle.
    """
    data = list(ws.values)
    if not data or len(data) < 2:
        print(f"[Cycle Analysis] Worksheet {ws.title} has no data.")
        return

    header_row = data[0]
    df = pd.DataFrame(data[1:], columns=header_row)

    # Robust column detection
    freq_col = next((col for col in df.columns if "frequency" in str(col).lower()), None)
    if "Rs" not in df.columns or "X" not in df.columns or freq_col is None:
        print(f"[Cycle Analysis] Worksheet {ws.title}: Missing 'Rs', 'X', or frequency column.")
        return

    freq = pd.to_numeric(df[freq_col], errors='coerce').to_numpy()
    cycles = split_cycles_from_frequency(freq)

    if not cycles:
        print(f"[Cycle Analysis] Worksheet {ws.title}: No cycles detected.")
        return

    analysis_list = []
    cycle_number = 1
    for start, end in cycles:
        print(f"[Cycle Analysis] Cycle {cycle_number}: rows {start} to {end}")
        cycle_df = df.iloc[start:end]
        if len(cycle_df) < 3:
            print(f"[Cycle Analysis] Skipping short cycle {cycle_number}")
            continue
        analysis = ac.compute_analysis(cycle_df)
        if analysis:
            analysis["Cycle"] = cycle_number
            analysis_list.append(analysis)
            cycle_number += 1

    if not analysis_list:
        print(f"[Cycle Analysis] Worksheet {ws.title}: No valid cycles for analysis.")
        return

    headers_out = [
        "Cycle",
        "Global Minima (X-Value)", "Global Minima (Y-Value)",
        "Curve First (X-Value)", "Curve First (Y-Value)",
        "Curve Last (X-Value)", "Curve Last (Y-Value)",
        "Linear First (X-Value)", "Linear First (Y-Value)",
        "Linear Last (X-Value)", "Linear Last (Y-Value)",
        "angle", "linear equation", "Whole Slope",
        "Slope 1", "Slope 2", "Slope 3", "Slope 4", "Slope 5",
        "Delta R_e", "Delta R_ct", "Delta R_int"
    ]
    start_col = 8
    for i, header in enumerate(headers_out):
        ws.cell(row=1, column=start_col + i, value=header)
    for row_num, analysis in enumerate(analysis_list, start=2):
        for i, header in enumerate(headers_out):
            ws.cell(row=row_num, column=start_col + i, value=analysis.get(header, None))

