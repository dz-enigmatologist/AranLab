import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from circle_fit import taubinSVD
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares
from sklearn.linear_model import LinearRegression, RANSACRegressor
import math


def remove_outliers_iqr(data):
    """Remove outliers using IQR method and return filtered data + indices kept"""
    data = np.array(data)
    if len(data) < 4:  # not enough points for IQR
        return data, np.arange(len(data))
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask], np.where(mask)[0]

def deepta_analysis_functions(df, cycle_idx, time_per_cycle, cp1, ph1, freq_array, Z_array, debug=False):
    """
    Main analysis function that processes EIS data and returns results.
    Includes circle fitting logic from old code implementation.
    """
    if debug:
        print(f"\n=== DEBUG: Starting analysis for cycle {cycle_idx} ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    results = {
        "time(mins)": time_per_cycle * cycle_idx,
        "Cp1": cp1,
        "Ph1": ph1,
        "Rct_semicircle": None,
        "Rs": None,
        "fit_success": False,
        "transition_index": None,
        "fit_quality": None,
        "failure_reason": None,
        "method_used": None,
    }

    # Convert and clean input
    x_raw = pd.to_numeric(df.get("Rs"), errors="coerce").to_numpy()
    y_raw = pd.to_numeric(df.get("X"), errors="coerce").to_numpy()
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]
    y_raw = np.abs(y_raw[mask])

    if len(x_raw) < 5:
        results["failure_reason"] = f"Insufficient data points ({len(x_raw)} < 5)"
        return results

    # Sort & unique
    sort_idx = np.argsort(x_raw)
    x_raw, y_raw = x_raw[sort_idx], y_raw[sort_idx]

    try:
        # --- Find transition point using the old method ---
        transition_idx = find_transition_point_optimized(x_raw, y_raw, debug=debug)
        results["transition_index"] = int(transition_idx)

        # --- Simple Plotting Section ---
        if debug:
            plt.figure(figsize=(10, 8))
            plt.scatter(x_raw, y_raw, label="Raw Data", color="black", s=15)
            
            # Add vertical line at transition
            if 0 <= transition_idx < len(x_raw):
                plt.axvline(x=x_raw[transition_idx], color="red", linestyle="--", 
                           label="Transition Point", alpha=0.7)
            
            plt.xlabel("Z' (Ohm)")
            plt.ylabel("Z'' (Ohm)")
            plt.title(f"Nyquist Plot - Cycle {cycle_idx}\nTransition Point Detection")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # --- OLD CODE INTEGRATION: Circle fitting with multiple shift attempts ---
        for shift in range(0, 501, 100):
            current_transition_idx = min(transition_idx + shift, len(x_raw) - 3)
            results["transition_index"] = int(current_transition_idx)
            
            x_fit_region = x_raw[current_transition_idx:]
            y_fit_region = y_raw[current_transition_idx:]

            if len(x_fit_region) < 3:
                if debug:
                    print(f"⚠️ Fit region too small ({len(x_fit_region)} pts), skipping shift {shift}")
                continue

            # === Method A: Circle Fit (TaubinSVD) - OLD CODE IMPLEMENTATION ===
            try:
                # CHANGE THIS: Use data from START to transition point instead of transition point to END
                x_fit_region = x_raw[:current_transition_idx]  # From start to transition point
                y_fit_region = y_raw[:current_transition_idx]  # From start to transition point
                
                preFiltered_x = x_fit_region
                preFiltered_y = y_fit_region

                # remove outliers
                y_fit_region, keep_idx = remove_outliers_iqr(preFiltered_y)       
                x_fit_region = preFiltered_x[keep_idx]

                # Prepare coordinates for circle fitting
                circle_coords = np.column_stack((x_fit_region, y_fit_region))
                
                # Fit circle using TaubinSVD (from old code)
                xc, yc, r, sigma = taubinSVD(circle_coords)
                
                # Plot the circle fit in debug mode (from old code)
                if debug:
                    plot_circle_fit(x_fit_region, y_fit_region, xc, yc, r, 
                                  shift, cycle_idx, debug=debug)
                
                # Validate the circle fit
                validation = validate_circle_fit(x_fit_region, y_fit_region, xc, yc, r, debug=debug)
                
                if validation["is_valid"]:
                    results["Rct_semicircle"] = 2.0 * r
                    results["Rs"] = xc - r
                    results["fit_success"] = True
                    results["fit_quality"] = sigma
                    results["method_used"] = "taubin_circle"
                    
                    if debug:
                        print(f"✅ TaubinSVD Circle fit successful: Rct={results['Rct_semicircle']:.3f}, Rs={results['Rs']:.3f}")
                    
                    # Plot final successful fit
                    if debug:
                        plot_final_circle_fit(x_raw, y_raw, x_fit_region, y_fit_region, 
                                            xc, yc, r, cycle_idx, shift)
                    
                    return results
                else:
                    if debug:
                        print(f"⚠️ TaubinSVD Circle fit failed validation: {validation['reason']}")
                        
            except Exception as e:
                if debug:
                    print(f"⚠️ TaubinSVD Circle fit exception: {e}")

            # === Method B: Randles Fit ===
            randles_results = fit_randles_circuit(freq_array, x_raw, y_raw, debug=debug)
            if randles_results:
                results.update(randles_results)
                results["fit_success"] = True
                results["method_used"] = "randles"
                if debug:
                    print("✅ Randles fit successful.")
                return results
            else:
                if debug:
                    print("⚠️ Randles fit failed.")

            # === Method C: Ellipse Fit ===
            try:
                ellipse_results = fit_ellipse(x_fit_region, y_fit_region, debug=debug)
                
                # Always plot the ellipse in debug mode, regardless of success
                if debug:
                    try:
                        # Try to get ellipse parameters for plotting even if fit_ellipse returned None
                        ellipse_params = {}
                        try:
                            # Replicate the ellipse fitting logic for plotting
                            X = np.column_stack((x_fit_region**2, x_fit_region*y_fit_region, y_fit_region**2, 
                                            x_fit_region, y_fit_region))
                            A = X.T @ X
                            C_mat = np.zeros((6,6))
                            C_mat[0,2] = -2; C_mat[1,1] = 1; C_mat[2,0] = -2
                            
                            eigvals, eigvecs = np.linalg.eig(np.linalg.inv(A) @ C_mat)
                            conic_coeffs = eigvecs[:, np.argmin(np.abs(eigvals-1))]
                            conic_coeffs /= conic_coeffs[-1]
                            
                            a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, f_coeff = conic_coeffs
                            det = b_coeff**2 - 4*a_coeff*c_coeff
                            
                            if det < 0:  # It's an ellipse
                                xc = (2*c_coeff*d_coeff - b_coeff*e_coeff) / det
                                yc = (2*a_coeff*e_coeff - b_coeff*d_coeff) / det
                                
                                a_sq = -2*(a_coeff*e_coeff**2+c_coeff*d_coeff**2-b_coeff*d_coeff*e_coeff+det*f_coeff)/(det*((b_coeff**2-4*a_coeff*c_coeff)-1))
                                b_sq = -2*(a_coeff*e_coeff**2+c_coeff*d_coeff**2-b_coeff*d_coeff*e_coeff+det*f_coeff)/(det*((b_coeff**2-4*a_coeff*c_coeff)+1))
                                
                                if a_sq > 0 and b_sq > 0:
                                    a_len = np.sqrt(a_sq)
                                    b_len = np.sqrt(b_sq)
                                    ellipse_params = {"xc": xc, "yc": yc, "a": a_len, "b": b_len}
                        except Exception:
                            pass  # If we can't calculate parameters, we'll skip detailed plotting
                        
                        # Create plot
                        #plt.figure(figsize=(8, 6))
                        plt.scatter(x_fit_region, y_fit_region, label="Fit Region Data", color="blue", s=15)
                        
                        if ellipse_params:  # If we have parameters, plot the ellipse
                            xc, yc, a_len, b_len = ellipse_params["xc"], ellipse_params["yc"], ellipse_params["a"], ellipse_params["b"]
                            
                            # Generate ellipse points
                            t = np.linspace(0, 2*np.pi, 100)
                            x_ellipse = xc + a_len * np.cos(t)
                            y_ellipse = yc + b_len * np.sin(t)
                            
                            plt.plot(x_ellipse, y_ellipse, 'm-', label=f"Fitted Ellipse (a={a_len:.1f}, b={b_len:.1f})", alpha=0.7)
                            plt.scatter([xc], [yc], color='orange', marker='x', s=50, label=f"Center ({xc:.1f}, {yc:.1f})")
                            
                            # Calculate and show Rct estimate
                            Rct_estimate = 2 * max(a_len, b_len)
                            plt.text(0.05, 0.95, f"Rct estimate: {Rct_estimate:.1f} Ω", 
                                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
                        
                        plt.xlabel("Z' (Ohm)")
                        plt.ylabel("Z'' (Ohm)")
                        plt.title(f"Ellipse Fit Attempt - Shift {shift}\nCycle {cycle_idx}")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.axis('equal')
                        plt.tight_layout()
                        plt.show()
                        
                        if ellipse_params:
                            print(f"📊 Ellipse fit plotted: center=({xc:.1f}, {yc:.1f}), axes=({a_len:.1f}, {b_len:.1f})")
                        else:
                            print("📊 Ellipse fit attempted but no valid ellipse parameters found")
                        
                    except Exception as plot_error:
                        print(f"⚠️ Failed to plot ellipse: {plot_error}")
                
                if ellipse_results:
                    results.update(ellipse_results)
                    results["fit_success"] = True
                    results["method_used"] = "ellipse"
                    if debug:
                        print("✅ Ellipse fit successful.")
                    return results
                else:
                    if debug:
                        print("⚠️ Ellipse fit failed.")
            except Exception as e:
                if debug:
                    print(f"⚠️ Ellipse fit exception: {e}")

            # === Method D: Least-Squares Circle Fit ===
            try:
                ls_circle_results = fit_least_squares_circle(x_fit_region, y_fit_region, debug=debug)
                
                # Always plot the least-squares circle in debug mode, regardless of success
                if debug:
                    try:
                        # Try to get circle parameters for plotting even if fit_least_squares_circle returned None
                        circle_params = {}
                        try:
                            # Replicate the least-squares fitting logic for plotting
                            Rs0 = np.min(x_fit_region)
                            Rct0 = np.ptp(x_fit_region) / 2.0
                            xc0, yc0, r0 = Rs0 + Rct0, 0, Rct0
                            
                            def circle_residuals(params, x, y):
                                xc, yc, r = params
                                return np.sqrt((x - xc)**2 + (y - yc)**2) - r

                            result = least_squares(circle_residuals, [xc0, yc0, r0], args=(x_fit_region, y_fit_region))
                            
                            if result.success:
                                xc_fit, yc_fit, r_fit = result.x
                                circle_params = {"xc": xc_fit, "yc": yc_fit, "r": r_fit, "cost": result.cost}
                        except Exception:
                            pass  # If we can't calculate parameters, we'll skip detailed plotting
                        
                        # Create plot
                        #plt.figure(figsize=(8, 6))
                        plt.scatter(x_fit_region, y_fit_region, label="Fit Region Data", color="blue", s=15)
                        
                        if circle_params:  # If we have parameters, plot the circle
                            xc, yc, r = circle_params["xc"], circle_params["yc"], circle_params["r"]
                            
                            # Generate circle points
                            theta = np.linspace(0, 2*np.pi, 100)
                            x_circle = xc + r * np.cos(theta)
                            y_circle = yc + r * np.sin(theta)
                            
                            plt.plot(x_circle, y_circle, 'c-', label=f"LS Fitted Circle (r={r:.1f})", alpha=0.7)
                            plt.scatter([xc], [yc], color='purple', marker='x', s=50, label=f"Center ({xc:.1f}, {yc:.1f})")
                            
                            # Show cost/error
                            cost = circle_params.get("cost", float('nan'))
                            plt.text(0.05, 0.95, f"LS Cost: {cost:.3e}", 
                                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
                            
                            # Calculate and show Rct estimate
                            Rct_estimate = 2 * r
                            plt.text(0.05, 0.88, f"Rct estimate: {Rct_estimate:.1f} Ω", 
                                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
                        
                        plt.xlabel("Z' (Ohm)")
                        plt.ylabel("Z'' (Ohm)")
                        plt.title(f"Least-Squares Circle Fit Attempt - Shift {shift}\nCycle {cycle_idx}")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.axis('equal')
                        plt.tight_layout()
                        plt.show()
                        
                        if circle_params:
                            print(f"📊 Least-squares circle fit plotted: center=({xc:.1f}, {yc:.1f}), radius={r:.1f}, cost={cost:.3e}")
                        else:
                            print("📊 Least-squares circle fit attempted but no valid parameters found")
                        
                    except Exception as plot_error:
                        print(f"⚠️ Failed to plot least-squares circle: {plot_error}")
                
                if ls_circle_results:
                    results.update(ls_circle_results)
                    results["fit_success"] = True
                    results["method_used"] = "ls_circle"
                    if debug:
                        print("✅ Least-squares circle fit successful.")
                    return results
                else:
                    if debug:
                        print("⚠️ Least-squares circle fit failed.")
            except Exception as e:
                if debug:
                    print(f"⚠️ Least-squares circle fit exception: {e}")

            # === Method E: Polynomial Fit (x^4) ===
            try:
                poly_results = fit_x4_poly(x_fit_region, y_fit_region, debug=debug)
                
                # Always plot the polynomial fit in debug mode, regardless of success
                if debug:
                    try:
                        # Try to get polynomial parameters for plotting even if fit_x4_poly returned None
                        poly_params = {}
                        try:
                            # Fit 4th order polynomial
                            poly_coeffs = np.polyfit(x_fit_region, y_fit_region, 4)
                            poly_params = {"coefficients": poly_coeffs}
                            
                            # Calculate Rct estimate if possible
                            roots = np.roots(poly_coeffs)
                            real_roots = roots[np.isreal(roots)].real
                            
                            if len(real_roots) >= 2:
                                real_roots.sort()
                                Rs_estimate = real_roots[-2]
                                Rct_estimate = real_roots[-1] - Rs_estimate
                                poly_params.update({"Rs_estimate": Rs_estimate, "Rct_estimate": Rct_estimate})
                                
                        except Exception:
                            pass  # If we can't calculate parameters, we'll skip detailed plotting
                        
                        # Create plot
                        #plt.figure(figsize=(8, 6))
                        plt.scatter(x_fit_region, y_fit_region, label="Fit Region Data", color="blue", s=15)
                        
                        if poly_params and "coefficients" in poly_params:  # If we have coefficients, plot the polynomial
                            coeffs = poly_params["coefficients"]
                            
                            # Generate polynomial curve
                            x_curve = np.linspace(min(x_fit_region), max(x_fit_region), 100)
                            y_curve = np.polyval(coeffs, x_curve)
                            
                            plt.plot(x_curve, y_curve, 'y-', label="4th Order Polynomial Fit", alpha=0.7, linewidth=2)
                            
                            # Show polynomial equation
                            eq_text = f"y = {coeffs[0]:.3e}x⁴ + {coeffs[1]:.3e}x³ + {coeffs[2]:.3e}x² + {coeffs[3]:.3e}x + {coeffs[4]:.3e}"
                            plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes, 
                                    bbox=dict(facecolor='white', alpha=0.8), fontsize=8)
                            
                            # Show Rct estimate if available
                            if "Rct_estimate" in poly_params and "Rs_estimate" in poly_params:
                                Rct_est = poly_params["Rct_estimate"]
                                Rs_est = poly_params["Rs_estimate"]
                                plt.text(0.05, 0.85, f"Rct estimate: {Rct_est:.1f} Ω\nRs estimate: {Rs_est:.1f} Ω", 
                                        transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
                            
                            # Mark the roots
                            roots = np.roots(coeffs)
                            real_roots = roots[np.isreal(roots)].real
                            y_roots = np.polyval(coeffs, real_roots)
                            plt.scatter(real_roots, y_roots, color='orange', marker='o', s=50, label="Polynomial Roots")
                        
                        plt.xlabel("Z' (Ohm)")
                        plt.ylabel("Z'' (Ohm)")
                        plt.title(f"4th Order Polynomial Fit Attempt - Shift {shift}\nCycle {cycle_idx}")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        
                        if poly_params and "coefficients" in poly_params:
                            print(f"📊 4th order polynomial fit plotted: {len(poly_params['coefficients'])-1}th order")
                            if "Rct_estimate" in poly_params:
                                print(f"   Rct estimate: {poly_params['Rct_estimate']:.1f} Ω")
                        else:
                            print("📊 Polynomial fit attempted but no valid parameters found")
                        
                    except Exception as plot_error:
                        print(f"⚠️ Failed to plot polynomial: {plot_error}")
                
                if poly_results:
                    results.update(poly_results)
                    results["fit_success"] = True
                    results["method_used"] = "poly_x4"
                    if debug:
                        print("✅ x^4 Polynomial fit successful.")
                    return results
                else:
                    if debug:
                        print("⚠️ x^4 Polynomial fit failed.")
            except Exception as e:
                if debug:
                    print(f"⚠️ x^4 Polynomial fit exception: {e}")

        # If we reach here, all methods failed
        results["failure_reason"] = "All fitting methods failed after shifting transition point."
        if debug:
            print("❌ All fitting methods failed after retries.")

    except Exception as e:
        results["failure_reason"] = f"Exception: {e}"
        if debug:
            import traceback
            traceback.print_exc()

    return results

def find_transition_point_optimized(x_in, y_in, debug=False,
                                   ransac_min_inlier_ratio=0.7,
                                   ransac_min_samples=3):
    """Multi-method robust transition point detection."""
    x = np.asarray(x_in).astype(float).ravel()
    y = np.asarray(y_in).astype(float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 5: return 0
    order = np.argsort(x)
    x = x[order]; y = y[order]
    ux, uidx = np.unique(x, return_index=True)
    x = x[uidx]; y = y[uidx]
    n = len(x)
    if n < 5: return 0
    if debug: print(f"find_transition_point_optimized: n={n}, x_range=({x[0]:.2f},{x[-1]:.2f})")
    candidates = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
        denom = (1 + dy**2)**1.5
        denom[~np.isfinite(denom)] = 1e-10
        curvature = np.abs(d2y) / denom
        curvature[~np.isfinite(curvature)] = 0.0
        c_idx = int(np.nanargmax(curvature))
        c_idx = max(1, min(c_idx, n-3))
        candidates.add(c_idx)
        if debug: print(f"  curvature_idx={c_idx}")
    except Exception as e:
        if debug: print("  curvature failed:", e)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
        mask = np.isfinite(d2y)
        if np.any(mask):
            d_idx = int(np.nanargmax(np.abs(d2y)))
        else:
            d_idx = int(np.argmin(y))
        d_idx = max(1, min(d_idx, n-3))
        candidates.add(d_idx)
        if debug: print(f"  derivative_idx={d_idx}")
    except Exception as e:
        if debug: print("  derivative failed:", e)
    try:
        m_idx = int(np.argmin(y))
        m_idx = max(1, min(m_idx, n-3))
        candidates.add(m_idx)
        if debug: print(f"  min_idx={m_idx}")
    except Exception as e:
        if debug: print("  min failed:", e)
    try:
        R = radius_of_curvature(x, y, debug=debug)
        finite = R[np.isfinite(R)]
        if finite.size:
            medianR = np.median(finite)
            k = 8.0 if np.nanmax(finite)/np.nanmin(finite) > 20 else 5.0
            thresh = medianR * k
            idxs = np.where(R > thresh)[0]
            if idxs.size:
                r_idx = int(idxs[0])
                r_idx = max(1, min(r_idx, n-3))
                candidates.add(r_idx)
                if debug: print(f"  radius_threshold_idx={r_idx}, thresh={thresh:.3e}")
    except Exception as e:
        if debug: print("  radius method failed:", e)
    try:
        min_tail = max(4, n // 6)
        found_ransac = None
        for tail_len in range(min_tail, n-2):
            k = n - tail_len
            xr = x[k:].reshape(-1, 1); yr = y[k:]
            ransac = RANSACRegressor(LinearRegression(), min_samples=min(0.5*len(yr), ransac_min_samples), residual_threshold=np.std(yr)*0.5 if np.std(yr)>0 else 1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try: ransac.fit(xr, yr)
                except Exception: continue
            inlier_mask = ransac.inlier_mask_
            inlier_ratio = np.sum(inlier_mask) / len(inlier_mask) if len(inlier_mask) else 0.0
            if inlier_ratio >= ransac_min_inlier_ratio:
                found_ransac = k
                break
        if found_ransac is not None:
            rr_idx = max(1, min(found_ransac, n-3))
            candidates.add(rr_idx)
            if debug: print(f"  ransac_tail_idx={rr_idx} (tail_len={n-rr_idx}, inlier_ratio>={ransac_min_inlier_ratio})")
    except Exception as e:
        if debug: print("  ransac failed:", e)
    if not candidates: candidates.add(int(np.argmin(y)))
    cand_list = sorted(list(candidates))
    if debug: print("  candidates:", cand_list)
    best_idx = None
    best_score = np.inf
    for c in cand_list:
        score = piecewise_sse_score(x, y, c, left_degree=2)
        if debug: print(f"    candidate {c} -> score {score:.6e}")
        if score < best_score:
            best_score = score
            best_idx = c
    if best_idx is None:
        if cand_list: best_idx = int(np.median(cand_list))
        else: best_idx = int(np.argmin(y))
    best_idx = max(1, min(best_idx, n-3))
    if debug: print(f"  chosen transition_idx={best_idx} with score={best_score:.6e}")
    return best_idx


# --- OLD CODE INTEGRATION: Helper functions ---

def find_transition_point_old_method(x, y, debug=False):
    """
    Find the transition point from semicircular to linear region using the old method.
    This replicates the logic from your old code.
    """
    found_min = 0
    notAsemicircle = 0
    transition_idx = 0
    
    # Old method: find where the curve starts increasing
    for index in range(0, len(x) - 1):
        if index == len(x) - 1:
            # If we haven't found the increasing point
            notAsemicircle = 1
            break
        if y[index] < y[index + 1]:  # When y starts increasing
            transition_idx = index
            break
    
    if debug:
        if notAsemicircle:
            print("⚠️ No clear semicircle transition found")
        else:
            print(f"📊 Transition point found at index {transition_idx}, x={x[transition_idx]:.1f}")
    
    return transition_idx

def plot_circle_fit(x, y, xc, yc, r, shift, cycle_idx, debug=False):
    """Plot circle fit with detailed visualization (from old code)"""
    try:
        #plt.figure(figsize=(12, 8))
        
        # Plot raw data
        plt.scatter(x, y, label="Fit Region Data", color="blue", s=20, alpha=0.7)
        
        # Plot fitted circle
        theta = np.linspace(0, 2 * math.pi, 100)
        x_circle = xc + r * np.cos(theta)
        y_circle = yc + r * np.sin(theta)
        plt.plot(x_circle, y_circle, 'r-', linewidth=2, 
                label=f"Fitted Circle (r={r:.1f} Ω)", alpha=0.8)
        
        # Mark center
        plt.scatter([xc], [yc], color='green', marker='x', s=100, 
                   label=f"Center ({xc:.1f}, {yc:.1f})")
        
        # Add diameter line
        plt.plot([xc - r, xc + r], [yc, yc], 'g--', alpha=0.6, 
                label=f"Diameter: {2*r:.1f} Ω")
        
        plt.xlabel("Z' (Ohm)")
        plt.ylabel("Z'' (Ohm)")
        plt.title(f"Circle Fit - Cycle {cycle_idx}, Shift {shift}\nRct ≈ {2*r:.1f} Ω")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        if debug:
            print(f"📊 Circle fit plotted: center=({xc:.1f}, {yc:.1f}), radius={r:.1f} Ω")
            
    except Exception as e:
        if debug:
            print(f"⚠️ Failed to plot circle: {e}")

def plot_final_circle_fit(x_raw, y_raw, x_fit, y_fit, xc, yc, r, cycle_idx, shift):
    """Plot final successful circle fit on top of all data"""
    #plt.figure(figsize=(12, 8))
    
    # Plot all raw data
    plt.scatter(x_raw, y_raw, label="All Data", color="gray", s=10, alpha=0.5)
    
    # Highlight fit region
    plt.scatter(x_fit, y_fit, label="Fit Region", color="blue", s=20, alpha=0.8)
    
    # Plot fitted circle
    theta = np.linspace(0, 2 * math.pi, 100)
    x_circle = xc + r * np.cos(theta)
    y_circle = yc + r * np.sin(theta)
    plt.plot(x_circle, y_circle, 'r-', linewidth=3, 
            label=f"Fitted Circle (Rct={2*r:.1f} Ω)")
    
    # Mark center and show parameters
    plt.scatter([xc], [yc], color='green', marker='x', s=100, 
               label=f"Center ({xc:.1f}, {yc:.1f})")
    
    plt.xlabel("Z' (Ohm)")
    plt.ylabel("Z'' (Ohm)")
    plt.title(f"Successful Circle Fit - Cycle {cycle_idx}\nRct = {2*r:.1f} Ω, Rs = {xc - r:.1f} Ω")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def validate_circle_fit(x, y, xc, yc, r, debug=False):
    """
    Validate circle fit using residuals and physical checks.
    """
    result = {"is_valid": False, "reason": None}
    try:
        # Calculate distances from points to circle center
        distances = np.sqrt((x - xc)**2 + (y - yc)**2)
        residuals = np.abs(distances - r)
        mean_error = np.mean(residuals)
        max_error = np.max(residuals)

        # Physical constraints for EIS semicircles
        valid_radius = (r > 0) and (r < np.max(x) * 2)
        valid_center_y = (abs(yc) < r * 0.5)  # Center should be near real axis
        good_fit_quality = (mean_error < r * 0.2) and (max_error < r * 0.5)
        
        if debug:
            print(f"📊 Circle fit validation: mean_error={mean_error:.3f}, max_error={max_error:.3f}")
            print(f"📊 Radius check: {valid_radius}, Center Y check: {valid_center_y}, Fit quality: {good_fit_quality}")

        if not valid_radius:
            result["reason"] = f"Invalid radius ({r:.3f})"
        elif not valid_center_y:
            result["reason"] = f"Center too far from real axis (yc={yc:.3f})"
        elif not good_fit_quality:
            result["reason"] = f"Poor fit quality (mean_err={mean_error:.3f})"
        else:
            result["is_valid"] = True

    except Exception as e:
        result["reason"] = f"Validation error: {e}"

    return result

def fit_least_squares_circle(x, y, debug=False):
    """Fit a circle using a least-squares optimization."""
    try:
        if len(x) < 3: return None
        
        # Initial guess from algebraic method or a simple midpoint guess
        Rs0 = np.min(x)
        Rct0 = np.ptp(x) / 2.0
        xc0, yc0, r0 = Rs0 + Rct0, 0, Rct0
        if debug:
            print(f"      LS Circle initial guess: center=({xc0:.3f}, {yc0:.3f}), r={r0:.3f}")
        
        # Objective function to minimize: sum of squared residuals
        def circle_residuals(params, x, y):
            xc, yc, r = params
            return np.sqrt((x - xc)**2 + (y - yc)**2) - r

        result = least_squares(circle_residuals, [xc0, yc0, r0], args=(x, y))
        
        if result.success:
            xc_fit, yc_fit, r_fit = result.x
            if validate_circle_fit(x, y, xc_fit, yc_fit, r_fit, debug=debug)["is_valid"]:
                return {"Rct_semicircle": 2.0 * r_fit, "Rs": xc_fit - r_fit, "fit_quality": result.cost}
        return None

    except Exception as e:
        if debug: print(f"      Least-squares circle fit failed: {e}")
        return None

def fit_x4_poly(x, y, debug=False):
    """Fit a 4th-order polynomial (x^4) to the data."""
    try:
        if len(x) < 5: return None
        
        # The form of the poly is ax^4 + bx^3 + cx^2 + dx + e
        poly_coeffs = np.polyfit(x, y, 4)
        
        # Find Rct by finding the roots where y=0
        roots = np.roots(poly_coeffs)
        real_roots = roots[np.isreal(roots)].real
        
        if len(real_roots) >= 2:
            real_roots.sort()
            Rs_fit = real_roots[-2]
            Rct_fit = real_roots[-1] - Rs_fit
            
            if Rct_fit > 0 and Rs_fit >= np.min(x) and Rs_fit <= np.max(x):
                # Calculate fit quality (SSE)
                y_pred = np.polyval(poly_coeffs, x)
                sse = np.sum((y - y_pred)**2)
                return {"Rct_poly": float(Rct_fit), "Rs_poly": float(Rs_fit), "fit_quality": sse}

        return None
        
    except Exception as e:
        if debug: print(f"      x^4 poly fit failed: {e}")
        return None

def fit_ellipse(x, y, debug=False):
    """Fit an ellipse to the data points."""
    try:
        if len(x) < 5: return None
        
        # Based on method by Halir and Flusser (1998)
        # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        X = np.column_stack((x**2, x*y, y**2, x, y))
        A = X.T @ X
        C = np.zeros((6,6))
        C[0,2] = -2; C[1,1] = 1; C[2,0] = -2
        
        # solve generalized eigenvalue problem
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(A) @ C)
        conic_coeffs = eigvecs[:, np.argmin(np.abs(eigvals-1))]
        conic_coeffs /= conic_coeffs[-1] # normalize for stability
        
        # Get ellipse parameters from conic section coefficients
        a,b,c,d,e,f = conic_coeffs
        det = b**2 - 4*a*c
        if det >= 0: return None # Not an ellipse
        
        xc = (2*c*d - b*e) / det
        yc = (2*a*e - b*d) / det
        
        # Get axis lengths
        a_sq = -2*(a*e**2+c*d**2-b*d*e+det*f)/(det*((b**2-4*a*c)-1))
        b_sq = -2*(a*e**2+c*d**2-b*d*e+det*f)/(det*((b**2-4*a*c)+1))
        
        if a_sq < 0 or b_sq < 0: return None # Invalid axis lengths
        a_len = np.sqrt(a_sq)
        b_len = np.sqrt(b_sq)
        
        # Rct is the major axis length
        Rct_fit = 2 * max(a_len, b_len)
        Rs_fit = xc - max(a_len, b_len)
        
        if Rct_fit > 0 and Rs_fit < np.max(x) and Rs_fit > np.min(x) - 0.5 * np.ptp(x):
            return {"Rct_ellipse": Rct_fit, "Rs_ellipse": Rs_fit}
        
        return None
    except Exception as e:
        if debug: print(f"      Ellipse fit failed: {e}")
        return None

def fit_randles_circuit(frequencies, Z_real, Z_imag_abs, debug=False):
    """Fit Randles circuit to (Z_real, Z_imag_abs)."""
    def randles_concat(f, Rs, Rct, Q, n):
        omega = 2 * np.pi * f
        # Zcpe in CPE parameterization: 1 / (Q*(j*omega)^n)
        jomega_n = (1j * omega) ** n
        Zcpe = 1.0 / (Q * jomega_n)
        Z = Rs + 1.0 / (1.0 / Rct + 1.0 / Zcpe)
        real = np.real(Z)
        imag = np.imag(Z)  # imag likely negative on Nyquist
        return np.concatenate([real, imag])

    try:
        # Ensure arrays are 1D and same length
        freqs = np.asarray(frequencies).ravel()
        Zr = np.asarray(Z_real).ravel()
        Zi_abs = np.asarray(Z_imag_abs).ravel()
        if not (len(freqs) == len(Zr) == len(Zi_abs)):
            if debug:
                print("       Randles: length mismatch")
            return None

        # Prepare concatenated target (real then imag)
        Z_target = np.concatenate([Zr, -Zi_abs])  # note: feeding negative imag for model consistency

        # initial guess (positive)
        Rs0 = max(1.0, np.min(Zr))
        Rct0 = max(1.0, np.ptp(Zr) * 0.5)
        Q0 = 1e-6
        n0 = 0.9
        p0 = [Rs0, Rct0, Q0, n0]

        # bounds: Rs >= 0, Rct >= 0, Q>0, 0<n<=1
        lower = [0.0, 0.0, 1e-12, 0.1]
        upper = [np.inf, np.inf, 1e-2, 1.0]

        if debug:
            print(f"       Randles initial guess: Rs={Rs0:.3f}, Rct={Rct0:.3f}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                lambda f_flat, Rs, Rct, Q, n: randles_concat(f_flat, Rs, Rct, Q, n),
                freqs, Z_target, p0=p0, bounds=(lower, upper), maxfev=20000
            )

        Rs_fit, Rct_fit, Q_fit, n_fit = popt
        return {"Rct_randles": float(Rct_fit), "Rs_randles": float(Rs_fit), "Q": float(Q_fit), "n": float(n_fit)}

    except Exception as e:
        if debug:
            print(f"       Randles fit failed: {e}")
        return None

def radius_of_curvature(x, y, debug=False):
    """Compute local radius of curvature R(x) = (1 + y'**2)^(3/2) / |y''|"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
    # avoid divide by zero
    denom = np.abs(d2y)
    denom[denom < 1e-12] = 1e-12
    R = (1.0 + dy**2)**1.5 / denom
    R[~np.isfinite(R)] = np.nan
    if debug:
        finite = R[np.isfinite(R)]
        if finite.size:
            print(f"      Radius_of_curvature range: [{np.nanmin(finite):.3e}, {np.nanmax(finite):.3e}]")
    return R

def piecewise_sse_score(x, y, idx, left_degree=2):
    """Fit polynomial on x[:idx] and linear on x[idx:], return normalized SSE."""
    n = len(x)
    if idx < 3 or idx > n-3:
        return np.inf

    # left fit (polynomial)
    xl = x[:idx]
    yl = y[:idx]
    xr = x[idx:]
    yr = y[idx:]
    try:
        # polynomial fit left
        p = np.polyfit(xl, yl, deg=min(left_degree, max(1, len(xl)-1)))
        pred_l = np.polyval(p, xl)
        # linear fit right
        lr = LinearRegression()
        lr.fit(xr.reshape(-1,1), yr)
        pred_r = lr.predict(xr.reshape(-1,1))
        sse = np.nansum((yl - pred_l)**2) + np.nansum((yr - pred_r)**2)
        return sse / (len(xl) + len(xr))
    except Exception:
        return np.inf

    """
    Plot various types of fits on the current active figure/axes.
    
    Parameters:
    x, y: Data points to plot
    method_name: Type of fit ('circle', 'polynomial', 'randles', 'linear_regression', 'exponential')
    fit_params: Dictionary of parameters needed for the specific fit type
    debug: If True, print detailed information about the plotting process
    """
    # Get current axes
    ax = plt.gca()
    
    if debug:
        print(f"🖌️  Plotting {method_name} fit with params: {fit_params}")
    
    try:
        if method_name == "linear_regression":
            # Check if we have the required parameters
            if "slope" not in fit_params or "intercept" not in fit_params:
                if debug:
                    print("⚠️  Missing parameters for linear regression: need 'slope' and 'intercept'")
                return
            
            slope = fit_params["slope"]
            intercept = fit_params["intercept"]
            
            # Generate line
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            
            # Plot the line
            ax.plot(x_line, y_line, 'b-', label="Linear fit", alpha=0.7)
            
            if debug:
                print(f"✅ Linear fit plotted: y = {slope:.3f}x + {intercept:.3f}")
                
        elif method_name == "exponential":
            # Check if we have the required parameters
            if "a" not in fit_params or "b" not in fit_params:
                if debug:
                    print("⚠️  Missing parameters for exponential fit: need 'a' and 'b'")
                return
            
            a = fit_params["a"]
            b = fit_params["b"]
            r2 = fit_params.get("r2", None)
            
            # Generate curve
            x_curve = np.linspace(min(x), max(x), 100)
            y_curve = a * np.exp(b * x_curve)
            
            # Create label with R² if available
            label = "Exponential fit"
            if r2 is not None:
                label = f"Exponential fit (R²={r2:.3f})"
            
            # Plot the curve
            ax.plot(x_curve, y_curve, 'r-', label=label, alpha=0.7)
            
            if debug:
                print(f"✅ Exponential fit plotted: y = {a:.3f} * exp({b:.3f}x)")
                if r2 is not None:
                    print(f"   R² = {r2:.3f}")
                    
        elif method_name == "circle":
            # Check if we have the required parameters
            if "xc" not in fit_params or "yc" not in fit_params or "r" not in fit_params:
                if debug:
                    print("⚠️  Missing parameters for circle fit: need 'xc', 'yc', and 'r'")
                return
            
            xc = fit_params["xc"]
            yc = fit_params["yc"]
            r = fit_params["r"]
            
            # Generate circle points
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = xc + r * np.cos(theta)
            y_circle = yc + r * np.sin(theta)
            
            # Plot the circle
            ax.plot(x_circle, y_circle, 'g-', label="Circle fit", alpha=0.7)
            
            if debug:
                print(f"✅ Circle fit plotted: center=({xc:.3f}, {yc:.3f}), radius={r:.3f}")
                
        elif method_name == "polynomial":
            # Check if we have the required parameters
            if "coefficients" not in fit_params:
                if debug:
                    print("⚠️  Missing parameters for polynomial fit: need 'coefficients'")
                return
            
            coefficients = fit_params["coefficients"]
            
            # Generate polynomial curve
            x_curve = np.linspace(min(x), max(x), 100)
            y_curve = np.polyval(coefficients, x_curve)
            
            # Plot the polynomial
            order = len(coefficients) - 1
            ax.plot(x_curve, y_curve, 'm-', label=f"Polynomial fit (order {order})", alpha=0.7)
            
            if debug:
                print(f"✅ Polynomial fit plotted: order {order}")
                
        elif method_name == "randles":
            # This would be more complex - for now just plot a placeholder
            if debug:
                print("ℹ️  Randles circuit fit visualization not yet implemented")
                
        else:
            if debug:
                print(f"⚠️  Unknown fit method: {method_name}")
                
    except Exception as e:
        if debug:
            print(f"❌ Error plotting {method_name} fit: {str(e)}")