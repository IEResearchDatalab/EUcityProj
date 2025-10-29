import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


def build_bs_basis(
    x: np.ndarray,
    knots_internal: np.ndarray,
    degree: int = 2,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Build a B-spline basis matrix using SciPy BSpline.

    We construct a knot vector `t` with length `ncoef + degree + 1` where
    `ncoef = degree + len(knots_internal)` to match R's df convention.

    To build the knot vector we use the following layout (for degree k):
      t = [lower repeated k times] + internal_knots + [upper repeated (k+1) times]

    This yields len(t) = k + len(internal_knots) + (k+1) = (degree + len(internal_knots)) + degree + 1
    which gives ncoef = degree + len(knots_internal) basis functions.

    Args:
        x: 1D array-like of evaluation points.
        knots_internal: list/array of internal knot locations.
        degree: spline degree (k).
        lower_bound, upper_bound: boundary knots (scalars). If None, use min/max(x).

    Returns:
        ndarray shape (len(x), ncoef) design matrix.
    """
    x = np.asarray(x)
    if lower_bound is None:
        lower_bound = float(np.min(x))
    if upper_bound is None:
        upper_bound = float(np.max(x))

    k = int(degree)
    internal = list(map(float, knots_internal))

    # Number of basis functions desired (matches R's vardf)
    ncoef = k + len(internal)

    # Construct knot vector t
    # Repeat lower_bound k times and upper_bound k+1 times (see note above)
    t = ([float(lower_bound)] * k) + internal + ([float(upper_bound)] * (k + 1))
    t = np.asarray(t, dtype=float)

    # Sanity: t must be non-decreasing
    if not np.all(np.diff(t) >= 0):
        raise ValueError("Knot vector must be non-decreasing")

    # Build each basis by using coefficient vectors that pick out each basis function
    B = np.zeros((len(x), ncoef), dtype=float)
    for j in range(ncoef):
        c = np.zeros(ncoef, dtype=float)
        c[j] = 1.0
        spline = BSpline(t, c, k, extrapolate=True)
        B[:, j] = spline(x)

    return B


def erf(tmean: np.ndarray, perc: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Exposure-Response Function matching the R implementation.

    Steps replicated from `03_attribution.R`:
      - build a bs basis with degree=2 and internal knots at the specified percentiles
      - compute linear predictor lp = B(x) @ coefs
      - find MMT as the temperature within 25-99th percentiles that minimizes lp
      - compute rr(x) = exp(lp(x) - lp(mmt)) and clip at >= 1

    Args:
        x: scalar or 1D array-like of temperatures (same units as the tmean_distribution)

    Returns:
        np.ndarray of RR values
    """
    t0 = float(tmean[np.where(perc == 0)[0][0]])
    t10 = float(tmean[np.where(perc == 10)[0][0]])
    t75 = float(tmean[np.where(perc == 75)[0][0]])
    t90 = float(tmean[np.where(perc == 90)[0][0]])
    t100 = float(tmean[np.where(perc == 100)[0][0]])

    # Build basis at the evaluation points using same knots/bounds as R
    knots_internal = np.array([t10, t75, t90])
    B = build_bs_basis(
        tmean, knots_internal, degree=2, lower_bound=t0, upper_bound=t100
    )

    # Linear predictor (R treats coefficients as log-RR coefficients)
    lp = B.dot(coeffs)

    # Evaluate lp on the percentile grid to find MMT within 25-99% (replicates R's ind)
    # We need lp evaluated on the same tmean/perc grid
    B_full = build_bs_basis(
        tmean, knots_internal, degree=2, lower_bound=t0, upper_bound=t100
    )
    lp_full = B_full.dot(coeffs)
    ind = (perc >= 25) & (perc <= 99)
    if ind.sum() == 0:
        # fallback: use overall min
        mmt = tmean[np.argmin(lp_full)]
    else:
        mmt = tmean[ind][np.argmin(lp_full[ind])]

    # lp at mmt
    B_mmt = build_bs_basis(
        np.array([mmt]), knots_internal, degree=2, lower_bound=t0, upper_bound=t100
    )
    # Extract scalar safely to avoid deprecated array->scalar conversion
    lp_mmt = (B_mmt.dot(coeffs)).item()

    # Compute RR centered at MMT and enforce >= 1 as in R: rr <- pmax(exp(...), 1)
    rr = np.exp(lp - lp_mmt)
    rr = np.maximum(rr, 1.0)

    return rr


def main():
    # Read input data
    df_coeffs = pd.read_csv("data/coefs.csv")
    df_tmean = pd.read_csv("data/tmean_distribution.csv")

    # Filter for a single example (kept from original script)
    urau_code = "AT001C"

    plt.figure(figsize=(8, 5))
    for agegroup in ["20-44", "45-64", "65-74", "75-84", "85+"]:
        coeffs = df_coeffs[
            (df_coeffs["URAU_CODE"] == urau_code) & (df_coeffs["agegroup"] == agegroup)
        ]
        coeffs = coeffs.drop(columns=["URAU_CODE", "agegroup"]).values[0]
        tmean = (
            df_tmean[df_tmean["URAU_CODE"] == urau_code]
            .drop(columns=["URAU_CODE"])
            .values[0]
        )
        perc = np.array(df_tmean.drop(columns=["URAU_CODE"]).columns.tolist())
        # Format: "x.x%" turn to float
        perc = np.array([float(pct.strip("%")) for pct in perc])

        y_vals = erf(tmean, perc, coeffs)

        # Plot ERF
        plt.plot(perc, y_vals, label=agegroup)
    plt.xlabel("Temperature percentile")
    plt.ylabel("Relative Risk")
    plt.title(f"Exposure-Response Function (ERF) for {urau_code}")
    plt.legend(title="Age Group")
    plt.grid()
    plt.savefig("erf_plot.png")


if __name__ == "__main__":
    main()
