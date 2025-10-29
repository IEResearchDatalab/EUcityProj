import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

coeffs = pd.read_csv("data/coefs.csv")
tmean = pd.read_csv("data/tmean_distribution.csv")

# Filter for a single example
urau_code = "AT001C"
agegroup = "85+"
coeffs = coeffs[(coeffs["URAU_CODE"] == urau_code) & (coeffs["agegroup"] == agegroup)]
coeffs = coeffs.drop(columns=["URAU_CODE", "agegroup"]).values[0]
temp = tmean[tmean["URAU_CODE"] == urau_code].drop(columns=["URAU_CODE"]).values[0]
perc = np.array(tmean.drop(columns=["URAU_CODE"]).columns.tolist())
# Format: "x.x%" turn to float
perc = np.array([float(pct.strip("%")) for pct in perc])

# Find the percentages related to the 10th, 75th and 90th percentiles
t0 = temp[np.where(perc == 0)[0][0]]
t10 = temp[np.where(perc == 10)[0][0]]
t75 = temp[np.where(perc == 75)[0][0]]
t90 = temp[np.where(perc == 90)[0][0]]
t100 = temp[np.where(perc == 100)[0][0]]

# Build the ERF function as a B-spline
knots = np.array([t0, t0, t10, t75, t90, t100, t100])


def erf(x):
    """Exposure-Response Function (ERF) for temperature and mortality.

    Args:
        x (float or array-like): Temperature percentiles.

    Returns:
        float or array-like: Relative risk corresponding to the PM2.5 concentration.
    """

    # Define B-spline basis functions
    spline = BSpline(knots, coeffs, k=2, extrapolate=True)

    # Calculate relative risk
    rr = spline(x)

    # Find the minimum relative risk
    mrr = rr[np.argmin(rr)]

    # Normalize relative risk to have a minimum of 1
    rr = rr + (1 - mrr)

    return rr


# Plotting the ERF for visualization
y_vals = erf(temp)

plt.figure(figsize=(8, 5))
plt.plot(perc, y_vals, label=f"ERF for URAU {urau_code}, Age {agegroup}")
plt.xlabel("Temperature percentile")
plt.ylabel("Relative Risk")
plt.title("Exposure-Response Function (ERF) for Temperature and Mortality")
plt.legend()
plt.grid()
plt.savefig("erf_plot.png")
