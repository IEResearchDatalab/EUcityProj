import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


def build_bspline(
    x: np.ndarray,
    knots_internal: np.ndarray,
    coeffs: np.ndarray,
    degree: int = 2,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a B-spline basis matrix using SciPy BSpline.

    We construct a knot vector `t` with length `ncoef + degree + 1` where
    `ncoef = degree + len(knots_internal)` to match R's df convention.

    To build the knot vector we use the following layout (for degree k):
      t = [lower repeated k times] + internal_knots + [upper repeated (k+1) times]

    This yields len(t) = k + len(internal_knots) + (k+1) =
    (degree + len(internal_knots)) + degree + 1, which gives
    ncoef = degree + len(knots_internal) basis functions.

    Parameters
    ----------
    x : np.ndarray
        1D array-like of evaluation points.
    knots_internal : np.ndarray
        Array of internal knot locations.
    coeffs : np.ndarray
        Array of spline coefficients.
    degree : int, optional
        Spline degree (k), by default 2.
    lower_bound : float or None, optional
        Lower boundary knot. If None, use min(x), by default None.
    upper_bound : float or None, optional
        Upper boundary knot. If None, use max(x), by default None.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A callable BSpline object that evaluates the spline at given points.

    Raises
    ------
    ValueError
        If knot vector is not non-decreasing.
    """
    if lower_bound is None:
        lower_bound = float(np.min(x))
    if upper_bound is None:
        upper_bound = float(np.max(x))

    k = int(degree)
    internal = list(map(float, knots_internal))

    # Construct knot vector t
    # Repeat lower_bound k times and upper_bound k+1 times (see note above)
    t = ([float(lower_bound)] * k) + internal + ([float(upper_bound)] * (k + 1))
    t = np.asarray(t, dtype=float)

    # Sanity: t must be non-decreasing
    if not np.all(np.diff(t) >= 0):
        raise ValueError("Knot vector must be non-decreasing")

    # Build each basis by using coefficient vectors that pick out each basis function
    return BSpline(t, coeffs, k, extrapolate=True)


def erf(
    tmean: np.ndarray,
    perc: np.ndarray,
    coeffs: np.ndarray,
    knots_internal: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Compute Exposure-Response Function matching Masselot et al. (2025) R implementation.

    Steps replicated from `03_attribution.R`:
      1. Build a B-spline basis with degree=2 and FIXED internal knots from historical data
      2. Compute linear predictor lp = B(x) @ coefs
      3. Find MMT as the temperature within 25-99th percentiles that minimizes lp
      4. Compute rr(x) = exp(lp(x) - lp(mmt)) and clip at >= 1

    Parameters
    ----------
    tmean : np.ndarray
        1D array of temperature evaluation points (°C).
    perc : np.ndarray
        1D array of percentiles corresponding to tmean.
    coeffs : np.ndarray
        1D array of spline coefficients for this age group.
    knots_internal : np.ndarray
        FIXED internal knot locations (at 10%, 75%, 90% of historical data).
    lower_bound : float
        FIXED boundary knot at 0% of historical data.
    upper_bound : float
        FIXED boundary knot at 100% of historical data.

    Returns
    -------
    np.ndarray
        Array of relative risk (RR) values, clipped at minimum of 1.0.

    Notes
    -----
    The knots are fixed per city and shared across all age groups, matching R's
    `argvar` parameter in the original implementation.
    """
    # Build basis at the evaluation points using FIXED knots from historical data
    # These knots are the same for all age groups (matching R's argvar)
    bspline = build_bspline(
        tmean,
        knots_internal,
        coeffs,
        degree=2,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    lp = bspline(tmean)

    # Evaluate lp on the percentile grid to find MMT within 25-99% (replicates R's ind)
    # We need lp evaluated on the same tmean/perc grid
    ind = (perc >= 25) & (perc <= 99)
    if ind.sum() == 0:
        # fallback: use overall min
        mmt = tmean[np.argmin(lp)]
    else:
        mmt = tmean[ind][np.argmin(lp[ind])]

    # lp at mmt
    lp_mmt = bspline(np.array([mmt]))

    # Compute RR centered at MMT and enforce >= 1 as in R: rr <- pmax(exp(...), 1)
    rr = np.exp(lp - lp_mmt)
    rr = np.maximum(rr, 1.0)

    return rr


def main(
    urau_code: str = "AT001C",
    coefs_path: str = "data/coefs.csv",
    tmean_path: str = "data/tmean_distribution.csv",
    output_dir: str = "output",
    figsize: tuple[float, float] = (12, 5),
    save_plot: bool = True,
):
    """Generate Exposure-Response Function plots for a specified city.

    This function creates dual plots showing the relationship between temperature
    and relative risk (RR) for different age groups, using both absolute temperature
    and percentile scales.

    Parameters
    ----------
    urau_code : str, optional
        Urban Audit city code (e.g., "AT001C" for Vienna), by default "AT001C".
    coefs_path : str, optional
        Path to CSV file containing spline coefficients, by default "data/coefs.csv".
    tmean_path : str, optional
        Path to CSV file containing temperature distributions, by default "data/tmean_distribution.csv".
    output_dir : str, optional
        Directory where output plot will be saved, by default "output".
    figsize : tuple[float, float], optional
        Figure size (width, height) in inches, by default (12, 5).
    save_plot : bool, optional
        Whether to save the plot to disk, by default True.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        The generated figure and array of axes.

    Notes
    -----
    The function generates two side-by-side plots:
    - Left: RR vs Temperature (°C) with percentile secondary axis
    - Right: RR vs Temperature percentile with temperature secondary axis

    Age groups included: 20-44, 45-64, 65-74, 75-84, 85+

    References
    ----------
    Masselot, P., Mistry, M.N., Rao, S. et al. Estimating future heat-related
    and cold-related mortality under climate change, demographic and adaptation
    scenarios in 854 European cities. Nat Med (2025).
    https://doi.org/10.1038/s41591-024-03452-2
    """
    # Read input data
    df_coeffs = pd.read_csv(coefs_path)
    df_tmean = pd.read_csv(tmean_path)

    # Extract the temperature distribution for this city (used for evaluation)
    tmean = (
        df_tmean[df_tmean["URAU_CODE"] == urau_code]
        .drop(columns=["URAU_CODE"])
        .values[0]
    )
    perc = np.array(df_tmean.drop(columns=["URAU_CODE"]).columns.tolist())
    # Format: "x.x%" turn to float
    perc = np.array([float(pct.strip("%")) for pct in perc])

    # CRITICAL: Extract FIXED knots ONCE per city (matching R's argvar)
    # In R (03_attribution.R lines 172-176):
    #   varknots <- tper[paste0(varper, ".0%")]  # where varper = c(10, 75, 90)
    #   varbound <- range(tper)
    #   argvar <- list(fun = varfun, degree = vardegree, knots = varknots, Bound = varbound)
    t0 = float(tmean[np.where(perc == 0)[0][0]])  # Boundary knot (lower)
    t10 = float(tmean[np.where(perc == 10)[0][0]])  # Internal knot 1
    t75 = float(tmean[np.where(perc == 75)[0][0]])  # Internal knot 2
    t90 = float(tmean[np.where(perc == 90)[0][0]])  # Internal knot 3
    t100 = float(tmean[np.where(perc == 100)[0][0]])  # Boundary knot (upper)

    knots_internal = np.array([t10, t75, t90])
    lower_bound = t0
    upper_bound = t100

    # Create two subplots: one for X = percentiles, one for X = temperatures
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for agegroup in ["20-44", "45-64", "65-74", "75-84", "85+"]:
        # Extract coefficients for this URAU_CODE and agegroup
        coeffs = df_coeffs[
            (df_coeffs["URAU_CODE"] == urau_code) & (df_coeffs["agegroup"] == agegroup)
        ]
        coeffs = coeffs.drop(columns=["URAU_CODE", "agegroup"]).values[0]

        # Compute ERF values using FIXED knots (same for all age groups)
        y_vals = erf(tmean, perc, coeffs, knots_internal, lower_bound, upper_bound)

        # Plot ERF
        axs[0].plot(tmean, y_vals, label=agegroup)
        axs[1].plot(perc, y_vals, label=agegroup)
    axs[0].set_xlabel("Temperature (ºC)")
    axs[1].set_xlabel("Temperature percentile")
    for ax in axs:
        ax.set_ylabel("Relative Risk")
        ax.set_title(f"Exposure-Response Function (ERF) for {urau_code}")
        ax.legend(title="Age Group")
        ax.grid()
    # Below the X-axis temperatures, create a secondary X-axis with percentiles
    # showing the percentiles 10, 25, 50, 75, 90
    secax = axs[0].secondary_xaxis(
        -0.15,
        functions=(
            lambda x: np.interp(
                x,
                tmean,
                perc,
            ),
            lambda x: np.interp(
                x,
                perc,
                tmean,
            ),
        ),
    )
    secax.set_xlabel("Temperature Percentile")
    secax.set_xticks([10, 25, 50, 75, 90])

    # And now the opposite on the percentile plot
    # We will show the temperatures corresponding to existing ticks
    secax2 = axs[1].secondary_xaxis(
        -0.15,
        functions=(
            lambda x: np.interp(
                x,
                perc,
                tmean,
            ),
            lambda x: np.interp(
                x,
                tmean,
                perc,
            ),
        ),
    )
    secax2.set_xlabel("Temperature (ºC)")
    xticks = axs[1].get_xticks()
    secax2.set_xticks(np.interp(xticks, perc, tmean).round(1))

    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"erf_plot_{urau_code}.png"), dpi=150)
        print(f"Plot saved to {os.path.join(output_dir, f'erf_plot_{urau_code}.png')}")

    return fig, axs


if __name__ == "__main__":
    main()
