import numpy as np

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

    return {
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
        "Delta R_ct": R_ct,
        "Delta R_int": R_int
    }

def add_analysis_to_sheet(ws, analysis, start_col=8):
    """
    Writes analysis headers in row 1 (starting at column H) and the analysis results in row 2.
    The mapping is as follows:
      H: Cycle              (for Excel cycle analysis only)
      I: Global Minima (X-Value)
      J: Global Minima (Y-Value)
      K: Curve First (X-Value)
      L: Curve First (Y-Value)
      M: Curve Last (X-Value)
      N: Curve Last (Y-Value)
      O: Linear First (X-Value)
      P: Linear First (Y-Value)
      Q: Linear Last (X-Value)
      R: Linear Last (Y-Value)
      S: angle
      T: linear equation
      U: Whole Slope
      V: Slope 1
      W: Slope 2
      X: Slope 3
      Y: Slope 4
      Z: Slope 5
      AA: R_e
      AB: R_ct
      AC: R_int
    For CSV sheets (not cycles) only columns H to AC (excluding "Cycle") are used.
    For Excel cycle analysis, a "Cycle" value is included.
    """
    # For CSV analysis, headers start at H.
    headers = [
        "Global Minima (X-Value)", "Global Minima (Y-Value)",
        "Curve First (X-Value)", "Curve First (Y-Value)",
        "Curve Last (X-Value)", "Curve Last (Y-Value)",
        "Linear First (X-Value)", "Linear First (Y-Value)",
        "Linear Last (X-Value)", "Linear Last (Y-Value)",
        "angle", "linear equation", "Whole Slope",
        "Slope 1", "Slope 2", "Slope 3", "Slope 4", "Slope 5",
        "Delta R_e", "Delta R_ct", "Delta R_int"
    ]
    # If a cycle value is present, prepend "Cycle" to headers.
    if "Cycle" in analysis:
        headers = ["Cycle"] + headers
    for i, header in enumerate(headers):
        ws.cell(row=1, column=start_col + i, value=header)
    for i, header in enumerate(headers):
        ws.cell(row=2, column=start_col + i, value=analysis.get(header, None))