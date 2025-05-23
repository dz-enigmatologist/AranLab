import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import re
from openpyxl.utils import get_column_letter
from collections import defaultdict

from  ME_modules.ME_impedance_py_modules.processing import analyze_cycle
from  ME_modules.ME_impedance_py_modules.visualization import plot_nyquist

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

def compute_analysis(df):
    """
    Computes analysis for a CSV file.
    
    Uses:
      - "Rs" as x-values.
      - Absolute value of "X" as y-values.
    
    The data is sorted by x-values.
    The global minimum is determined only from points with x >= 10000.
    
    Analysis details:
      - "Curve First" is the first sorted point.
      - "Curve Last" and "Linear First" are set as the global minimum (from x>=10000).
      - "Linear Last" is the last sorted point.
      - The linear region (from the global minimum to the end) is divided into 5 segments to compute slopes.
      - An overall linear regression on the linear region yields:
          * Whole Slope,
          * linear equation (y = m*x + b),
          * angle (degrees) of the regression line.
      - Resistances:
          R_e = first x,
          R_ct = global_min_x - first x,
          R_int = last x - global_min_x.
    
    Returns a dictionary with keys:
      "Global Minima (X-Value)", "Global Minima (Y-Value)",
      "Curve First (X-Value)", "Curve First (Y-Value)",
      "Curve Last (X-Value)", "Curve Last (Y-Value)",
      "Linear First (X-Value)", "Linear First (Y-Value)",
      "Linear Last (X-Value)", "Linear Last (Y-Value)",
      "angle", "linear equation", "Whole Slope",
      "Slope 1", "Slope 2", "Slope 3", "Slope 4", "Slope 5",
      "Delta R_e", "Delta R_ct", "Delta R_int"
    """
    try:
        x_raw = df["Rs"].to_numpy(dtype=float)
        y_raw = np.abs(df["X"].to_numpy(dtype=float))
    except KeyError:
        print("Required columns 'Rs' and 'X' not found in CSV.")
        return None

    if len(x_raw) < 5:
        return None

    # Save unsorted "Curve First" point
    first_x = x_raw[0]
    print(f"first_x: {first_x}")
    first_y = y_raw[0]
    print(f"first_y: {first_y}")

    # Sort by x
    sort_idx = np.argsort(x_raw)
    x_sorted = x_raw[sort_idx]
    y_sorted = y_raw[sort_idx]

    # Find global min restricted to x >= 10000
    valid_indices = np.where(x_sorted >= 10000)[0]
    if len(valid_indices) == 0:
        idx = int(np.argmin(y_sorted))
    else:
        start_idx = valid_indices[0]
        subset_min_idx = int(np.argmin(y_sorted[start_idx:]))
        idx = start_idx + subset_min_idx

    global_min_x = x_sorted[idx]
    global_min_y = y_sorted[idx]

    # Define key points
    last_x = x_sorted[-1]
    print(f"last_x: {last_x}")
    last_y = y_sorted[-1]
    print(f"last_y: {last_y}")

    # Linear region: from global min to end
    x_linear = x_sorted[idx:]
    y_linear = y_sorted[idx:]
    n_linear = len(x_linear)

    # Compute segmented slopes
    slopes = []
    if n_linear < 5:
        slopes = [float('nan')] * 5
    else:
        segment_length = n_linear // 5
        for i in range(5):
            start_i = i * segment_length
            end_i = n_linear if i == 4 else (i + 1) * segment_length
            if end_i - start_i < 2:
                slopes.append(float('nan'))
            else:
                x_seg = x_linear[start_i:end_i]
                y_seg = y_linear[start_i:end_i]
                slope = max(0, np.polyfit(x_seg, y_seg, 1)[0])  # Clamp to positive
                slopes.append(slope)

    print(f"Linear region length: {n_linear}")
    print(f"x_linear: {x_linear}")
    print(f"y_linear: {y_linear}")

    # Whole-region linear fit
    if len(x_linear) >= 2:
        whole_fit = np.polyfit(x_linear, y_linear, 1)
        whole_slope = whole_fit[0]
        intercept = whole_fit[1]
        angle = np.degrees(np.arctan(whole_slope))
        linear_eq = f"y = {whole_slope:.3f}x + {intercept:.3f}"
    else:
        whole_slope = float('nan')
        angle = float('nan')
        linear_eq = None

    # Resistances
    R_e = first_x
    R_ct = global_min_x - first_x
    R_int = last_x - global_min_x

    print("Returning slopes:", slopes)

    analysis = {
        "Global Minima (X-Value)": global_min_x,
        "Global Minima (Y-Value)": global_min_y,
        "Curve First (X-Value)": first_x,
        "Curve First (Y-Value)": first_y,
        "Curve Last (X-Value)": global_min_x,
        "Curve Last (Y-Value)": global_min_y,
        "Linear First (X-Value)": global_min_x,
        "Linear First (Y-Value)": global_min_y,
        "Linear Last (X-Value)": last_x,
        "Linear Last (Y-Value)": last_y,
        "angle": angle,
        "linear equation": linear_eq,
        "Whole Slope": whole_slope,
        "Slope 1": slopes[0],
        "Slope 2": slopes[1],
        "Slope 3": slopes[2],
        "Slope 4": slopes[3],
        "Slope 5": slopes[4],
        "Delta R_e": R_e,
        "Delta R_ct-a": R_ct,
        "Delta R_int": R_int
    }
    
    print("Analysis for CSV:", analysis)
    

    return analysis

# Full column structure, including placeholders
RESULT_COLUMNS = [
    'time(s)', 'delta Rct-a', 'Rct-d', 'Cp1', 'Ph1',
    'Slope 1', 'Slope 2', 'Slope 3', 'Slope 4', 'Slope 5',
    'Angle', 'Cp_exp-a', 'Cp_exp-b', 'Ph_slope', 'Ph_peak',
    'Area Cp', 'Area Ph', 'Area Slope', 'Area Rs-direct', 'Area Rs-Para',
    '', '',  # Placeholders
    'Cycle', 'Model', 'Rs', 'Rp', 'Q', 'n'
]

def normalize_sheet_name(raw_name):
    """
    Convert raw sheet names to standardized format using comprehensive mappings.
    """
    # First check for last wash sheets
    if "last wash" in raw_name.lower():
        return None
    
    # Comprehensive mapping dictionary
    MAPPINGS = {
        # 0 concentration
        "0pM_asso": ['0 pM_Cas_only_Association', '0 pM_Cas_complex_Association', 
                    '0 Cas only_Association', '0 Cas complex_Association'],
        "0pM_disso": ['0 pM_Cas_only_Dissociation', '0 pM_Cas_complex_Dissociation',
                     '0 Cas only_Dissociation', '0 Cas complex_Dissociation'],
        
        # 100 pM
        "100pM_asso": ['100 pM_Cas_only_Association', '100 pM_Cas_complex_Association',
                      '100 pM Cas only_Association', '100 pM Cas complex_Association'],
        "100pM_disso": ['100 pM_Cas_only_Dissociation', '100 pM_Cas_complex_Dissociation',
                       '100 pM Cas only_Dissociation', '100 pM Cas complex_Dissociation'],
        
        # 1 nM
        "1nM_asso": ['1 nM_Cas_only_Association', '1 nM_Cas_complex_Association',
                    '1 nM Cas only_Association', '1 nM Cas complex_Association'],
        "1nM_disso": ['1 nM_Cas_only_Dissociation', '1 nM_Cas_complex_Dissociation',
                     '1 nM Cas only_Dissociation', '1 nM Cas complex_Dissociation'],
        
        # 10 nM
        "10nM_asso": ['10 nM_Cas_only_Association', '10 nM_Cas_complex_Association',
                     '10 nM Cas only_Association', '10 nM Cas complex_Association'],
        "10nM_disso": ['10 nM_Cas_only_Dissociation', '10 nM_Cas_complex_Dissociation',
                      '10 nM Cas only_Dissociation', '10 nM Cas complex_Dissociation'],
        
        # 100 nM
        "100nM_asso": ['100 nM_Cas_only_Association', '100 nM_Cas_complex_Association',
                      '100 nM Cas only_Association', '100 nM Cas complex_Association'],
        "100nM_disso": ['100 nM_Cas_only_Dissociation', '100 nM_Cas_complex_Dissociation',
                       '100 nM Cas only_Dissociation', '100 nM Cas complex_Dissociation'],
        
        # Special cases
        "step_asso": ["Association step"],
        "step_disso": ["Dissociation step"]
    }
    
    # Create forward lookup
    forward_map = {}
    for norm_name, variants in MAPPINGS.items():
        for variant in variants:
            forward_map[variant] = norm_name
    
    # Check exact match first
    if raw_name in forward_map:
        return forward_map[raw_name]
    
    # Check case-insensitive and space-insensitive match
    clean_name = re.sub(r'\s+', '', raw_name).lower()
    for variant, norm_name in forward_map.items():
        if re.sub(r'\s+', '', variant).lower() == clean_name:
            return norm_name
    
    return None 

def process_chip_excel_only(chip_number: str, file_paths: dict) -> list:    
    """
    Process Excel file and return structured results.
    """
    if "cas" not in file_paths:
        print(f"[Chip {chip_number}] No CAS file found.")
        return []

    try:
        wb = load_workbook(file_paths["cas"], data_only=True)
    except Exception as e:
        print(f"[Chip {chip_number}] Failed to open workbook: {e}")
        return []

    results = []

    for raw_sheet_name in wb.sheetnames:
        if "last wash" in raw_sheet_name.lower():
            continue

        sheet_name = normalize_sheet_name(raw_sheet_name)
        if not sheet_name:
            continue

        try:
            conc, phase = sheet_name.split("_")
        except ValueError:
            continue

        # Load sheet data
        target_sheet = wb[raw_sheet_name]
        data = list(target_sheet.values)
        if not data or len(data) < 2:
            print(f"[Chip {chip_number}] Sheet '{raw_sheet_name}' is empty or missing headers.")
            continue

        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)

        # Validate required columns
        if "Rs" not in df.columns or "X" not in df.columns:
            print(f"[Chip {chip_number}] Sheet '{raw_sheet_name}' missing required columns.")
            continue

        # Get frequency data
        try:
            frequency_col = next(c for c in df.columns if "frequency" in str(c).lower())
            freq_array = pd.to_numeric(df[frequency_col], errors='coerce').to_numpy()
            if np.isnan(freq_array).any():
                print(f"[Chip {chip_number}] Invalid frequency values in sheet '{raw_sheet_name}'")
                continue
        except StopIteration:
            print(f"[Chip {chip_number}] Sheet '{raw_sheet_name}' missing frequency column.")
            continue

        # Create complex impedance
        Z = df["Rs"].to_numpy() + 1j * (-np.abs(df["X"].to_numpy()))

        # Split into cycles
        cycles = split_cycles_from_frequency(freq_array)
        if not cycles:
            print(f"[Chip {chip_number}] No valid cycles in sheet '{raw_sheet_name}'.")
            continue

        for idx, (start, end) in enumerate(cycles, start=1):
            cycle_freq = freq_array[start:end]
            cycle_Z = Z[start:end]
            cycle_df = df.iloc[start:end]

            if len(cycle_df) < 3:
                continue

            # Perform both traditional analysis and circuit fitting
            analysis = compute_analysis(cycle_df)
            fit_result = analyze_cycle(cycle_freq, cycle_Z, idx)

            if not analysis and not fit_result:
                continue

            # Prepare output row
            output_row = {col: None for col in RESULT_COLUMNS}
            
            # Traditional analysis results
            if analysis:
                row_data = {
                    'Slope 1': analysis.get("Slope 1"),
                    'Slope 2': analysis.get("Slope 2"),
                    'Slope 3': analysis.get("Slope 3"),
                    'Slope 4': analysis.get("Slope 4"),
                    'Slope 5': analysis.get("Slope 5"),
                    'Angle': analysis.get("angle"),
                    'Delta Rct-a': analysis.get("Delta R_ct-a"),
                    'Rct-d': None
                }
                for k, v in row_data.items():
                    output_row[k] = v

            # Circuit fitting results
            if fit_result:
                fit_data = {
                    'Model': fit_result['model'],
                    'Rs': fit_result['Rs'],
                    'Rp': fit_result['Rp'],
                    'Q': fit_result.get('Q'),
                    'n': fit_result.get('n'),
                    'C': fit_result.get('C'),
                }
                # Update output_row with fit_data
                for k, v in fit_data.items():
                    if k in output_row:  # Only update existing keys
                        output_row[k] = v

            # Update common fields with type-safe approach
            common_fields = {
                'Cycle': idx,
                'time(s)': None,  # Placeholder for time data if needed
                'Cp1': None,       # Placeholder for capacitance if needed
                'Ph1': None        # Placeholder for phase if needed
            }
            for key, value in common_fields.items():
                output_row[key] = value

            # Append the complete result
            results.append({
                "chip": chip_number,
                "concentration": conc,
                "phase": phase,
                "cycle": idx,
                "columns": output_row
            })

    print(f"[Chip {chip_number}] Collected {len(results)} valid analysis cycles.")
    return results

def write_chip_results_to_workbook(result_list, processed_chips_folder):
    """
    Writes results using normalized sheet names.
    Creates sheets if they don't exist.
    """
    from collections import defaultdict
    from openpyxl import Workbook
    
    # Organize by chip
    chip_data = defaultdict(list)
    for result in result_list:
        chip_data[result["chip"]].append(result)

    for chip, rows in chip_data.items():
        filename = f"Chip {chip}.xlsx"
        filepath = os.path.join(processed_chips_folder, filename)

        # Initialize workbook
        if os.path.exists(filepath):
            wb = load_workbook(filepath)
        else:
            wb = Workbook()
            # Remove default sheet if it exists
            if "Sheet" in wb.sheetnames:
                wb.remove(wb["Sheet"])

        for result in rows:
            sheet_name = f"{result['concentration']}_{result['phase']}"
            
            # Create sheet if it doesn't exist
            if sheet_name not in wb.sheetnames:
                new_sheet = wb.create_sheet(title=sheet_name)
                print(f"[Write] Created new sheet: {sheet_name}")
                # Write headers to new sheet
                for col_idx, col_name in enumerate(RESULT_COLUMNS, 1):
                    new_sheet.cell(row=1, column=col_idx, value=col_name)
            
            ws = wb[sheet_name]

            # Write data
            next_row = ws.max_row + 1
            for col_idx, col_name in enumerate(RESULT_COLUMNS, 1):
                ws.cell(row=next_row, column=col_idx, value=result['columns'].get(col_name))

        try:
            wb.save(filepath)
            print(f"[Write] Saved {len(rows)} rows to {filename}")
        except Exception as e:
            print(f"[Write] Failed to save {filename}: {str(e)}")