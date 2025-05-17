from .core import *
from .visualization import *

def analyze_cycle(frequencies, Z, cycle_num):
    """Analyze a single EIS cycle"""
    f_clean, Z_clean = prepare_impedance_data(frequencies, Z)
    
    # Try CPE model first
    try:
        popt, Z_fit = fit_randles_cpe(f_clean, Z_clean)
        return {
            'model': 'CPE',
            'parameters': popt,
            'Rs': popt[0], 'Rp': popt[1], 'Q': popt[2], 'n': popt[3],
            'frequencies': f_clean, 'Z': Z_clean, 'Z_fit': Z_fit
        }
    except Exception as e:
        print(f"CPE fit failed: {e}")
    
    # Fall back to RC model
    try:
        popt, Z_fit = fit_randles_rc(f_clean, Z_clean)
        return {
            'model': 'RC',
            'parameters': popt,
            'Rs': popt[0], 'Rp': popt[1], 'C': popt[2],
            'frequencies': f_clean, 'Z': Z_clean, 'Z_fit': Z_fit
        }
    except Exception as e:
        print(f"RC fit failed: {e}")
        return None

def process_eis_dataframe(df):
    """Process complete EIS dataset from DataFrame"""
    frequencies = df["Frequency(Hz)"].values.astype(float)
    Z = df["Rs"].values.astype(float) + 1j * (-np.abs(df["X"].values.astype(float)))
    
    cycles = split_cycles(frequencies)
    results = []
    
    for i, (start, end) in enumerate(cycles):
        result = analyze_cycle(frequencies[start:end], Z[start:end], i)
        if result:
            results.append(result)
            plot_nyquist(result['Z'], result['Z_fit'], f'Cycle {i+1} - {result["model"]} Fit')
    
    return results