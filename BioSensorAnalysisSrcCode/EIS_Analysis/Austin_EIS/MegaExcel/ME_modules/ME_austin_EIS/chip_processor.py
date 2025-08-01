import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from openpyxl import load_workbook, Workbook
from typing import Any


from ME_modules.ME_impedance_py_modules.processing import analyze_cycle

# --------------- CONFIG ---------------
all_raw_names_logged = []

RESULT_COLUMNS = [
    'time(mins)', 'delta Rct-a', 'Rct-d', 'Cp1', 'Ph1',
    'Slope 1', 'Slope 2', 'Slope 3', 'Slope 4', 'Slope 5',
    'Angle', 'Cp_exp-a', 'Cp_exp-b', 'Ph_slope', 'Ph_peak',
    'Area Cp', 'Area Ph', 'Area Slope', 'Area Rs-direct', 'Area Rs-Para',
    '', '',  # Placeholders
    'linear_eq_m', 'linear_eq_b', 'Rs', 'delta Rct-i', 'Q', 'n'
]

# --------------- FILE & SHEET HELPERS ---------------

def load_excel_safely(file_path, chip_number):
    try:
        return load_workbook(file_path, data_only=True)
    except Exception as e:
        print(f"[Chip {chip_number}] Failed to open workbook: {e}")
        return None

def should_skip_sheet(sheet_name):
    return "last wash" in sheet_name.lower() or not normalize_sheet_name(sheet_name)

def extract_sheet_data(wb, raw_sheet_name, chip_number):
    sheet = wb[raw_sheet_name]
    data = list(sheet.values)
    if not data or len(data) < 2:
        print(f"[Chip {chip_number}] Sheet '{raw_sheet_name}' is empty or missing headers.")
        return None
    headers = data[0]
    df = pd.DataFrame(data[1:], columns=headers)
    if not {"Rs", "X"}.issubset(df.columns):
        print(f"[Chip {chip_number}] Sheet '{raw_sheet_name}' missing required columns.")
        return None
    try:
        freq_col = next(c for c in df.columns if "frequency" in str(c).lower())
        freq_array = pd.to_numeric(df[freq_col], errors='coerce').to_numpy()
        if np.isnan(freq_array).any():
            print(f"[Chip {chip_number}] Invalid frequency values in sheet '{raw_sheet_name}'")
            return None
    except StopIteration:
        print(f"[Chip {chip_number}] Sheet '{raw_sheet_name}' missing frequency column.")
        return None
    Z = df["Rs"].to_numpy() + 1j * (-np.abs(df["X"].to_numpy()))
    return freq_array, Z, df

def extract_conc_phase(raw_name):
    normalized = normalize_sheet_name(raw_name)
    if not normalized or "_" not in normalized:
        return None, None
    return normalized.split("_", 1)

# --------------- CORE PROCESSING ---------------

def process_chip_excel_only(chip_number, file_paths):
    if "cas" not in file_paths:
        print(f"[Chip {chip_number}] No CAS file found.")
        return []

    wb = load_excel_safely(file_paths["cas"], chip_number)
    if wb is None:
        return []

    results = []

    for raw_sheet_name in wb.sheetnames:
        if should_skip_sheet(raw_sheet_name):
            continue

        sheet_data = extract_sheet_data(wb, raw_sheet_name, chip_number)
        if sheet_data is None:
            continue

        freq_array, Z_array, df = sheet_data
        cycles = split_cycles_from_frequency(freq_array)
        if not cycles:
            print(f"[Chip {chip_number}] No valid cycles in sheet '{raw_sheet_name}'.")
            continue

        conc, phase = extract_conc_phase(raw_sheet_name)
        if phase == "asso":
            total_time = 15 #minutes
        elif phase == "disso":
            total_time = 30 #minutes

        time_per_cycle = total_time / len(cycles)

        cp_cols = [col for col in df.columns if "cp" in col.lower()]
        ph_cols = [col for col in df.columns if "ph" in col.lower()]

        for idx, (start, end) in enumerate(cycles, start=1):
            cycle_df = df.iloc[start:end]
            if len(cycle_df) < 3:
                continue

            cp1 = cycle_df[cp_cols[0]].iloc[0] if cp_cols else None
            ph1 = cycle_df[ph_cols[0]].iloc[0] if ph_cols else None

            analysis_row = compute_analysis(
                cycle_df, idx, time_per_cycle, cp1, ph1, freq_array[start:end], Z_array[start:end]
            )

            if analysis_row is None:
                continue

            results.append({
                "chip": chip_number,
                "concentration": conc,
                "phase": phase,
                "cycle": idx,
                "columns": analysis_row
            })

    print(f"[Chip {chip_number}] Collected {len(results)} valid analysis cycles.")
    return results


def write_chip_results_to_workbook(result_list, processed_chips_folder):
    from collections import defaultdict
    chip_data = defaultdict(list)
    for result in result_list:
        chip_data[result["chip"]].append(result)

    for chip, rows in chip_data.items():
        filename = f"Chip {chip}.xlsx"
        filepath = os.path.join(processed_chips_folder, filename)
        wb = load_workbook(filepath) if os.path.exists(filepath) else Workbook()

        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])

        for result in rows:
            sheet_name = f"{result['concentration']}_{result['phase']}"
            if sheet_name not in wb.sheetnames:
                ws = wb.create_sheet(title=sheet_name)
                for col_idx, col_name in enumerate(RESULT_COLUMNS, 1):
                    ws.cell(row=1, column=col_idx, value=col_name)
                print(f"[Write] Created new sheet: {sheet_name}")

            ws = wb[sheet_name]
            next_row = ws.max_row + 1
            for col_idx, col_name in enumerate(RESULT_COLUMNS, 1):
                ws.cell(row=next_row, column=col_idx, value=result['columns'].get(col_name))

        try:
            wb.save(filepath)
            print(f"[Write] Saved {len(rows)} rows to {filename}")
        except Exception as e:
            print(f"[Write] Failed to save {filename}: {str(e)}")


# --------------- ANALYSIS LOGIC ---------------

def compute_analysis(df, cycle_idx, time_per_cycle, cp1, ph1, freq_array, Z_array): 
    try:
        x_raw = df["Rs"].to_numpy(dtype=float)
        y_raw = np.abs(df["X"].to_numpy(dtype=float))
    except KeyError:
        print("Required columns 'Rs' and 'X' not found in CSV.")
        return None

    if len(x_raw) < 2:
        return None

    
    sort_idx = np.argsort(x_raw)
    x_sorted = x_raw[sort_idx]
    y_sorted = y_raw[sort_idx]

    first_x = x_sorted[0]

    valid_indices = np.where(x_sorted >= 10000)[0]
    if len(valid_indices) > 0:
        min_region_start = valid_indices[0]
        idx_min_y = min_region_start + np.argmin(y_sorted[min_region_start:])
    else:
        idx_min_y = np.argmin(y_sorted)

    global_min_x = x_sorted[idx_min_y]

    x_linear = x_sorted[idx_min_y:]
    y_linear = y_sorted[idx_min_y:]
    n_linear = len(x_linear)

    slopes = [float('nan')] * 5
    if n_linear >= 5:
        segment_length = n_linear // 5
        for i in range(5):
            start_i = i * segment_length
            end_i = (i + 1) * segment_length if i < 4 else n_linear
            if end_i - start_i >= 2:
                slope, _ = np.polyfit(x_linear[start_i:end_i], y_linear[start_i:end_i], 1)
                slopes[i] = max(0, slope)
    elif n_linear >= 2:
        slope, _ = np.polyfit(x_linear, y_linear, 1)
        slopes[0] = max(0, slope)

    linear_eq_m = b = angle_val = float('nan')
    if len(x_linear) >= 2:
        linear_eq_m, b = np.polyfit(x_linear, y_linear, 1)
        angle_val = np.degrees(np.arctan(linear_eq_m))

    # Analyze the cycle with EIS fitting
    Rs = Rp = Q = n = None
    try:
        fit_result = analyze_cycle(freq_array, Z_array, cycle_idx)
        if fit_result:
            Rs = fit_result.get("Rs")
            Rp = fit_result.get("Rp")
            Q = fit_result.get("Q")
            n = fit_result.get("n")
    except Exception as e:
        print(f"[Cycle {cycle_idx}] analyze_cycle failed: {e}")
    
    #print(f"linear_eq_m = {linear_eq_m}")
    return {
        **{col: None for col in RESULT_COLUMNS},
        "time(mins)": time_per_cycle * cycle_idx,
        "Cp1": cp1,
        "Ph1": ph1,
        "delta Rct-a": global_min_x - first_x,
        "Slope 1": slopes[0],
        "Slope 2": slopes[1],
        "Slope 3": slopes[2],
        "Slope 4": slopes[3],
        "Slope 5": slopes[4],
        "Angle": angle_val,
        "linear_eq_m": linear_eq_m,
        "linear_eq_b": b,
        "Rs": Rs,
        "delta Rct-i": Rp,
        "Q": Q,
        "n": n
    }



def split_cycles_from_frequency(frequency, start_val=100, end_val=200000, tol_start=20, tol_end=10000):
    cycles = []
    in_cycle = False
    start_idx = None
    for i, freq in enumerate(frequency):
        if not in_cycle and abs(freq - start_val) <= tol_start:
            in_cycle = True
            start_idx = i
        elif in_cycle and abs(freq - end_val) <= tol_end:
            cycles.append((start_idx, i + 1))
            in_cycle = False
    return cycles

def normalize_sheet_name(raw_name):
    # Log the raw name as before
    all_raw_names_logged.append(raw_name)

    MAPPINGS = {
        "0pM_asso": ['0 pM_Cas_only_Association', '0 pM_Cas_complex_Association', '0 Cas only_Association', '0 Cas complex_Association'],
        "0pM_disso": ['0 pM_Cas_only_Dissociation', '0 pM_Cas_complex_Dissociation', '0 Cas only_Dissociation', '0 Cas complex_Dissociation'],
        "100pM_asso": ['100 pM_Cas_only_Association', '100 pM_Cas_complex_Association', '100 pM Cas only_Association', '100 pM Cas complex_Association'],
        "100pM_disso": ['100 pM_Cas_only_Dissociation', '100 pM_Cas_complex_Dissociation', '100 pM Cas only_Dissociation', '100 pM Cas complex_Dissociation'],
        "1nM_asso": ['1 nM_Cas_only_Association', '1 nM_Cas_complex_Association', '1 nM Cas only_Association', '1 nM Cas complex_Association'],
        "1nM_disso": ['1 nM_Cas_only_Dissociation', '1 nM_Cas_complex_Dissociation', '1 nM Cas only_Dissociation', '1 nM Cas complex_Dissociation'],
        "10nM_asso": ['10 nM_Cas_only_Association', '10 nM_Cas_complex_Association', '10 nM Cas only_Association', '10 nM Cas complex_Association'],
        "10nM_disso": ['10 nM_Cas_only_Dissociation', '10 nM_Cas_complex_Dissociation', '10 nM Cas only_Dissociation', '10 nM Cas complex_Dissociation'],
        "100nM_asso": ['100 nM_Cas_only_Association', '100 nM_Cas_complex_Association', '100 nM Cas only_Association', '100 nM Cas complex_Association'],
        "100nM_disso": ['100 nM_Cas_only_Dissociation', '100 nM_Cas_complex_Dissociation', '100 nM Cas only_Dissociation', '100 nM Cas complex_Dissociation'],
        "step_asso": ["Association step"], "step_disso": ["Dissociation step"]
    }

    # Pre-process MAPPINGS to create the forward_map with cleaned keys
    # This makes the lookup more robust.
    forward_map = {}
    for norm_name, variants in MAPPINGS.items():
        for variant in variants:
            # Clean the variant name by removing non-alphanumeric characters and lowercasing
            cleaned_variant = re.sub(r'[^a-z0-9]', '', variant.lower())
            forward_map[cleaned_variant] = norm_name

    # --- Step 1: Direct lookup (after cleaning raw_name) ---
    # Clean the input raw_name by removing non-alphanumeric characters and lowercasing
    cleaned_raw_name = re.sub(r'[^a-z0-9]', '', raw_name.lower())

    if cleaned_raw_name in forward_map:
        return forward_map[cleaned_raw_name]

    # If no match found after cleaning, return None
    return None
