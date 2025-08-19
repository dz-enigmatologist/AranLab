import numpy as np
import pandas as pd
import warnings
from circle_fit import taubinSVD
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares
from sklearn.linear_model import LinearRegression, RANSACRegressor

def deepta_analysis_functions(df, cycle_idx, time_per_cycle, cp1, ph1, freq_array, Z_array, debug=False):
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
    unique_x, unique_idx = np.unique(x_raw, return_index=True)
    x_raw, y_raw = x_raw[unique_idx], y_raw[unique_idx]

    try:
        # --- Step 1: Find initial transition point ---
        base_transition_idx = find_transition_point_optimized(x_raw, y_raw, debug=debug)

        # --- Step 2: Try fitting with multiple shifted transition points ---
        for shift in range(0, 501, 100):  # 0, 100, 200, 300, 400, 500
            transition_idx = min(base_transition_idx + shift, len(x_raw) - 3)
            results["transition_index"] = int(transition_idx)

            if debug:
                print(f"\n🔄 Retrying fit with transition index shifted by {shift} -> {transition_idx}")

            x_fit_region = x_raw[transition_idx:]
            y_fit_region = y_raw[transition_idx:]

            if len(x_fit_region) < 3:
                if debug:
                    print(f"⚠️ Fit region too small ({len(x_fit_region)} pts), skipping.")
                continue

            # === Method A: Circle Fit (TaubinSVD) ===
            try:
                circle_coords = np.column_stack((x_fit_region, y_fit_region))
                xc, yc, r, sigma = taubinSVD(circle_coords)
                validation = validate_circle_fit(x_fit_region, y_fit_region, xc, yc, r, debug=debug)
                if validation["is_valid"]:
                    results["Rct_semicircle"] = 2.0 * r
                    results["Rs"] = xc - r
                    results["fit_success"] = True
                    results["fit_quality"] = sigma
                    results["method_used"] = "taubin_circle"
                    if debug:
                        print(f"✅ TaubinSVD Circle fit successful: Rct={results['Rct_semicircle']:.3f}, Rs={results['Rs']:.3f}")
                    return results
                else:
                    if debug:
                        print(f"⚠️ TaubinSVD Circle fit failed: {validation['reason']}")
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
            ellipse_results = fit_ellipse(x_fit_region, y_fit_region, debug=debug)
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

            # === Method D: Least-Squares Circle Fit ===
            ls_circle_results = fit_least_squares_circle(x_fit_region, y_fit_region, debug=debug)
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

            # === Method E: Polynomial Fit (x^4) ===
            poly_results = fit_x4_poly(x_fit_region, y_fit_region, debug=debug)
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

        # If we reach here, everything failed even after shifting 500 pts
        results["failure_reason"] = "All fitting methods failed after shifting transition point."
        if debug:
            print("❌ All fitting methods failed after retries.")

    except Exception as e:
        results["failure_reason"] = f"Exception: {e}"
        if debug:
            import traceback
            traceback.print_exc()

    return results

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

def validate_circle_fit(x, y, xc, yc, r, debug=False):
    """Validate circle fit using residuals and simple physical checks."""
    result = {"is_valid": False, "reason": None}
    try:
        distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        residuals = np.abs(distances - r)
        mean_error = np.nanmean(residuals)

        if debug:
            print(f"      Circle mean residual: {mean_error:.6f}, r={r:.3f}")

        # basic checks
        valid_radius = (r > 0) and (r < max(1.0, np.max(x) * 4.0))
        valid_center = np.isfinite(xc) and np.isfinite(yc)

        # ratio-based tolerance for residual: require residual < 0.25*r
        good_fit = (mean_error < 0.25 * max(r, 1.0))

        if not valid_center:
            result["reason"] = f"Invalid center (xc or yc not finite)"
        elif not valid_radius:
            result["reason"] = f"Invalid radius ({r:.3f}, max_x={np.max(x):.3f})"
        elif not good_fit:
            result["reason"] = f"Poor fit quality (mean_err={mean_error:.3f}, r={r:.3f})"
        else:
            result["is_valid"] = True

    except Exception as e:
        result["reason"] = f"Validation exception: {e}"

    return result

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

def find_transition_point_optimized(x_in, y_in, debug=False,
                                   ransac_min_inlier_ratio=0.7,
                                   ransac_min_samples=3,
                                   smooth_window=5, smooth_poly=2):
    """Multi-method robust transition point detection with optional smoothing."""
    x = np.asarray(x_in).astype(float).ravel()
    y = np.asarray(y_in).astype(float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 5: return 0

    # Optional Savitzky-Golay smoothing
    from scipy.signal import savgol_filter
    if smooth_window >= 3 and smooth_window % 2 == 1 and x.size > smooth_window:
        y_smooth = savgol_filter(y, smooth_window, smooth_poly)
    else:
        y_smooth = y.copy()

    order = np.argsort(x)
    x = x[order]; y_smooth = y_smooth[order]; y = y[order]

    ux, uidx = np.unique(x, return_index=True)
    x = x[uidx]; y_smooth = y_smooth[uidx]; y = y[uidx]
    n = len(x)
    if n < 5: return 0

    if debug:
        print(f"find_transition_point_optimized: n={n}, x_range=({x[0]:.2f},{x[-1]:.2f})")

    candidates = set()
    try:
        # Curvature method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dy = np.gradient(y_smooth, x)
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
        # Second derivative method
        dy = np.gradient(y_smooth, x)
        d2y = np.gradient(dy, x)
        mask = np.isfinite(d2y)
        if np.any(mask):
            d_idx = int(np.nanargmax(np.abs(d2y)))
        else:
            d_idx = int(np.argmin(y_smooth))
        d_idx = max(1, min(d_idx, n-3))
        candidates.add(d_idx)
        if debug: print(f"  derivative_idx={d_idx}")
    except Exception as e:
        if debug: print("  derivative failed:", e)

    try:
        # Minimum y method
        m_idx = int(np.argmin(y_smooth))
        m_idx = max(1, min(m_idx, n-3))
        candidates.add(m_idx)
        if debug: print(f"  min_idx={m_idx}")
    except Exception as e:
        if debug: print("  min failed:", e)

    try:
        # Radius-of-curvature threshold method
        R = radius_of_curvature(x, y_smooth, debug=debug)
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
        # RANSAC tail method
        min_tail = max(4, n // 6)
        found_ransac = None
        for tail_len in range(min_tail, n-2):
            k = n - tail_len
            xr = x[k:].reshape(-1, 1)
            yr = y_smooth[k:]
            min_samples_int = max(1, int(min(0.5*len(yr), ransac_min_samples)))
            ransac = RANSACRegressor(
                LinearRegression(),
                min_samples=min_samples_int,
                residual_threshold=np.std(yr)*0.5 if np.std(yr) > 0 else 1e-6
            )
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

    if not candidates: candidates.add(int(np.argmin(y_smooth)))
    cand_list = sorted(list(candidates))
    if debug: print("  candidates:", cand_list)

    # Choose best candidate based on piecewise SSE
    best_idx = None
    best_score = np.inf
    for c in cand_list:
        score = piecewise_sse_score(x, y_smooth, c, left_degree=2)
        if debug: print(f"    candidate {c} -> score {score:.6e}")
        if score < best_score:
            best_score = score
            best_idx = c

    if best_idx is None:
        best_idx = int(np.median(cand_list)) if cand_list else int(np.argmin(y_smooth))
    best_idx = max(1, min(best_idx, n-3))
    if debug: print(f"  chosen transition_idx={best_idx} with score={best_score:.6e}")
    return best_idx
