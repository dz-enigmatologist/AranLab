from .core import *
from .visualization import *

def analyze_cycle(frequencies: np.ndarray, Z: np.ndarray, cycle_num: int) -> dict | None:
    """Analyze a single EIS cycle"""
    """
    Rs (Series Resistance / Solution Resistance / Ohmic Resistance) What it represents: 
        This is the resistance of the electrolyte solution between the working and reference 
        electrodes, as well as the resistance of the electrical connections, wires, and sometimes 
        the bulk material of the electrodes themselves. It's often the high-frequency intercept 
        on a Nyquist plot.

    Rp (Polarization Resistance / Charge Transfer Resistance) What it represents: 
        In a simple Randles equivalent circuit (which is a common model), Rp (often denoted as 
        Rct for charge transfer resistance) represents the resistance to the flow of charge 
        (electrons or ions) across the electrode-electrolyte interface. This is associated 
        with the kinetics of the electrochemical reaction occurring at the surface.

    Q (Constant Phase Element - CPE parameter Y0) What it represents: Q represents a Constant 
    Phase Element (CPE), which is a non-ideal capacitor. In real electrochemical systems, 
    the double-layer capacitance often behaves non-ideally due to factors like electrode surface 
    roughness, inhomogeneity, porous electrodes, or current distribution effects. A CPE describes 
    this non-ideal capacitive behavior.
    
    n (Constant Phase Element - CPE exponent) What it represents: This is the exponent of the 
    Constant Phase Element (CPE). It describes the "deviation from ideal behavior."
    
    """
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

def process_eis_dataframe(df: pd.DataFrame) -> list[dict]:
    """Process an EIS DataFrame into cycles and fit models.
    
    The DataFrame must have columns: 'Frequency(Hz)', 'Rs' (real part), and 'X' (imaginary part).
    Each cycle is split based on frequency patterns and analyzed with the CPE model, falling back to RC if necessary.
    Plots Nyquist plots for each cycle.
    
    Returns:
        A list of dicts with model fit results for each cycle.
    """

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