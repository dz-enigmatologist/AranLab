import matplotlib.pyplot as plt

def plot_nyquist(Z, Z_fit=None, title=None, ax=None):
    """Plot Nyquist plot of impedance data"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(Z.real, -Z.imag, 'o', label='Data')
    if Z_fit is not None:
        ax.plot(Z_fit.real, -Z_fit.imag, '-', label='Fit')
    
    ax.set_xlabel('Z\' (Ω)')
    ax.set_ylabel('-Z\'\' (Ω)')
    if title:
        ax.set_title(title)
    ax.legend()
    return ax

def plot_all_cycles(cycle_results):
    """Plot all cycles together"""
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, result in enumerate(cycle_results):
        ax.plot(result['Z'].real, -result['Z'].imag, 'o', markersize=4, label=f'Cycle {i+1}')
    ax.set_xlabel('Z\' (Ω)')
    ax.set_ylabel('-Z\'\' (Ω)')
    ax.set_title('All Cycles Overlay')
    ax.legend()
    return fig