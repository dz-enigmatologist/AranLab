import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from openpyxl import load_workbook, Workbook

from ME_modules.ME_impedance_py_modules.processing import analyze_cycle

# --------------- CONFIG ---------------

RESULT_COLUMNS = [
    'time(s)', 'delta Rct-a', 'Rct-d', 'Cp1', 'Ph1',
    'Slope 1', 'Slope 2', 'Slope 3', 'Slope 4', 'Slope 5',
    'Angle', 'Cp_exp-a', 'Cp_exp-b', 'Ph_slope', 'Ph_peak',
    'Area Cp', 'Area Ph', 'Area Slope', 'Area Rs-direct', 'Area Rs-Para',
    '', '',  # Placeholders
    'Cycle', 'Model', 'Rs', 'Rp', 'Q', 'n'
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
        total_time = 30  # Total experiment duration
        time_per_cycle = total_time / len(cycles) if len(cycles) > 0 else None

        for idx, (start, end) in enumerate(cycles, start=1):
            cycle_data = df.iloc[start:end]
            if len(cycle_data) < 3:
                continue

            # Get Cp1 and Ph1 (first values of columns containing "Cp" and "Ph")
            cp1, ph1 = None, None
            cp_cols = [col for col in df.columns if "cp" in col.lower()]
            ph_cols = [col for col in df.columns if "ph" in col.lower()]
            if cp_cols:
                cp1 = cycle_data[cp_cols[0]].iloc[0]
            if ph_cols:
                ph1 = cycle_data[ph_cols[0]].iloc[0]

            time_value = time_per_cycle * idx if time_per_cycle else None
            analysis = compute_analysis(cycle_data)
            fit_result = analyze_cycle(freq_array[start:end], Z_array[start:end], idx)

            if not analysis and not fit_result:
                continue

            output_row = build_result_row(analysis, fit_result, idx, time_value, cp1, ph1)
            results.append({
                "chip": chip_number,
                "concentration": conc,
                "phase": phase,
                "cycle": idx,
                "columns": output_row
            })
    print(f"[Chip {chip_number}] Collected {len(results)} valid analysis cycles.")
    return results


def build_result_row(analysis, fit_result, idx, time_value, cp1, ph1):
    row = {col: None for col in RESULT_COLUMNS}
    if analysis:
        for key in row:
            if key in analysis:
                row[key] = analysis[key]
    if fit_result:
        for key, val in fit_result.items():
            if key in row:
                row[key] = val
    row.update({
        "Cycle": idx,
        "time(s)": time_value,
        "Cp1": cp1,
        "Ph1": ph1
    })
    return row


def write_chip_results_to_workbook(result_list, processed_chips_folder):
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

def compute_analysis(df):
    try:
        x_raw = df["Rs"].to_numpy(dtype=float)
        y_raw = np.abs(df["X"].to_numpy(dtype=float))
    except KeyError:
        print("Required columns 'Rs' and 'X' not found in CSV.")
        return None
    if len(x_raw) < 2:
        return None
    first_x, first_y = x_raw[0], y_raw[0]
    sort_idx = np.argsort(x_raw)
    x_sorted, y_sorted = x_raw[sort_idx], y_raw[sort_idx]
    valid_indices = np.where(x_sorted >= 10000)[0]
    idx_min_y_region = valid_indices[0] + np.argmin(y_sorted[valid_indices[0]:]) if len(valid_indices) > 0 else np.argmin(y_sorted)
    global_min_x, global_min_y = x_sorted[idx_min_y_region], y_sorted[idx_min_y_region]
    last_x, last_y = x_sorted[-1], y_sorted[-1]
    x_linear, y_linear = x_sorted[idx_min_y_region:], y_sorted[idx_min_y_region:]
    n_linear = len(x_linear)
    slopes = [float('nan')] * 5
    if n_linear >= 5:
        segment_length = n_linear // 5
        for i in range(5):
            start_i, end_i = i * segment_length, (i + 1) * segment_length if i < 4 else n_linear
            if end_i - start_i >= 2:
                slopes[i] = max(0, np.polyfit(x_linear[start_i:end_i], y_linear[start_i:end_i], 1)[0])
    elif n_linear >= 2:
        slopes[0] = max(0, np.polyfit(x_linear, y_linear, 1)[0])
    linear_eq, angle_val = "y = nan*x + nan", float('nan')
    if len(x_linear) >= 2:
        m, b = np.polyfit(x_linear, y_linear, 1)
        angle_val = np.degrees(np.arctan(m))
        linear_eq = f"y = {m:.3f}x + {b:.3f}"
    return {
        "time(s)": None, "delta Rct-a": global_min_x - first_x, "Rct-d": None, "Cp1": None, "Ph1": None,
        "Slope 1": slopes[0], "Slope 2": slopes[1], "Slope 3": slopes[2], "Slope 4": slopes[3], "Slope 5": slopes[4],
        "Angle": angle_val, "Cp_exp-a": None, "Cp_exp-b": None, "Ph_slope": None, "Ph_peak": None,
        "Area Cp": None, "Area Ph": None, "Area Slope": None, "Area Rs-direct": None, "Area Rs-Para": None,
        "": None, " ": None, "Cycle": None, "Model": linear_eq, "Rs": first_x, "Rp": last_x - global_min_x, "Q": None, "n": None
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
    forward_map = {variant: norm_name for norm_name, variants in MAPPINGS.items() for variant in variants}
    if raw_name in forward_map:
        return forward_map[raw_name]
    clean_name = re.sub(r'\s+', '', raw_name).lower()
    for variant, norm_name in forward_map.items():
        if re.sub(r'\s+', '', variant).lower() == clean_name:
            return norm_name
    return None
