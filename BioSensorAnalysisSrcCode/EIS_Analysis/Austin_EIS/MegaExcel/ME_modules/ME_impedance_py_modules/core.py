import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def split_cycles(frequency, start_val=100, end_val=200000, tol_start=20, tol_end=10000):
    """Split frequency array into cycles with relaxed tolerances"""
    cycles = []
    in_cycle = False
    start_idx = None
    
    for i, freq in enumerate(frequency):
        if not in_cycle and abs(freq - start_val) <= tol_start:
            in_cycle = True
            start_idx = i + 1  # Skip first point
        elif in_cycle and abs(freq - end_val) <= tol_end:
            cycles.append((start_idx, i + 1))
            in_cycle = False
            
    if in_cycle:  # Handle incomplete cycle
        cycles.append((start_idx if start_idx is not None else i, len(frequency)))
    
    return cycles

def prepare_impedance_data(frequencies, Z):
    """Clean and sort impedance data"""
    valid_idx = (~np.isnan(Z.real)) & (~np.isnan(Z.imag)) & (frequencies > 0)
    f_clean = frequencies[valid_idx]
    Z_clean = Z[valid_idx]
    sorted_idx = np.argsort(f_clean)[::-1]  # High to low frequency
    return f_clean[sorted_idx], Z_clean[sorted_idx]

def fit_randles_cpe(f, Z):
    """Fit Randles circuit with CPE"""
    def model(f, R0, R1, Q, n):
        w = 2 * np.pi * f
        Z_CPE = 1/(Q * (1j * w)**n)
        return R0 + 1/(1/R1 + 1/Z_CPE)
    
    Z_stack = np.vstack([Z.real, Z.imag]).T
    popt, _ = curve_fit(
        lambda f, *p: np.vstack([model(f, *p).real, model(f, *p).imag]).T.flatten(),
        f, Z_stack.flatten(),
        p0=[np.median(Z.real)/2, np.median(Z.real)*2, 1e-5, 0.8],
        bounds=([0, 0, 1e-8, 0.5], [np.inf, np.inf, 1, 1]),
        maxfev=10000
    )
    return popt, model(f, *popt)

def fit_randles_rc(f, Z):
    """Fit simple Randles RC circuit"""
    def model(f, R0, R1, C):
        w = 2 * np.pi * f
        return R0 + R1/(1 + 1j * w * R1 * C)
    
    Z_stack = np.vstack([Z.real, Z.imag]).T
    popt, _ = curve_fit(
        lambda f, *p: np.vstack([model(f, *p).real, model(f, *p).imag]).T.flatten(),
        f, Z_stack.flatten(),
        p0=[np.median(Z.real)/2, np.median(Z.real)*2, 1e-5],
        bounds=([0, 0, 1e-8], [np.inf, np.inf, 1]),
        maxfev=10000
    )
    return popt, model(f, *popt)