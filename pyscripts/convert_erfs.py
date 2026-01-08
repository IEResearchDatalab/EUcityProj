"""
Extract RR_for_slopes_results and convert to a clean pandas DataFrame
"""

import os

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
rds_path = os.path.join(project_dir, "output", "erfs.rds")
output_dir = os.path.join(project_dir, "output")

print("Reading RDS file...")
data = r.readRDS(rds_path)

# Extract RR_for_slopes_results (the ERF data)
print("Extracting RR_for_slopes_results...")
rr_data = data.rx2("RR_for_slopes_results")

# Get all countries
countries = list(r["names"](rr_data))
print(f"Countries: {len(countries)}")

# Flatten into a DataFrame
records = []

for country in countries:
    country_data = rr_data.rx2(country)
    cities = list(r["names"](country_data))

    for city in cities:
        city_data = country_data.rx2(city)
        age_groups = list(r["names"](city_data))

        for age_group in age_groups:
            age_data = city_data.rx2(age_group)

            # Convert R dataframe to pandas
            with localconverter(robjects.default_converter + pandas2ri.converter):
                df = robjects.conversion.rpy2py(age_data)

            # Ensure it's a DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            # Add identifiers
            df = df.copy()
            df.insert(0, "country", country)
            df.insert(1, "city", city)
            df.insert(2, "age_group", age_group)

            records.append(df)

# Concatenate all records
print("Creating flat DataFrame...")
df_flat = pd.concat(records, ignore_index=True)

print(f"\nDataFrame shape: {df_flat.shape}")
print(f"Columns: {df_flat.columns.tolist()}")
print("\nFirst few rows:")
print(df_flat.head(10))

print("\nSummary by country:")
print(df_flat.groupby("country").size())

# Save in multiple formats
print("\n" + "=" * 60)
print("Saving files...")
print("=" * 60)

# 1. CSV
csv_path = os.path.join(output_dir, "erfs_rr_for_slopes.csv")
df_flat.to_csv(csv_path, index=False)
print(f"✓ CSV: {csv_path}")
print(f"  Size: {os.path.getsize(csv_path) / 1024:.2f} KB")

# 2. Pickle
pickle_path = os.path.join(output_dir, "erfs_rr_for_slopes.pkl")
df_flat.to_pickle(pickle_path)
print(f"✓ Pickle: {pickle_path}")
print(f"  Size: {os.path.getsize(pickle_path) / 1024:.2f} KB")

# 3. Try Parquet if available
try:
    parquet_path = os.path.join(output_dir, "erfs_rr_for_slopes.parquet")
    df_flat.to_parquet(parquet_path, index=False)
    print(f"✓ Parquet: {parquet_path}")
    print(f"  Size: {os.path.getsize(parquet_path) / 1024:.2f} KB")
except ImportError:
    print("⚠ Parquet not available (install pyarrow or fastparquet)")

print("\n" + "=" * 60)
print("Usage examples:")
print("=" * 60)
print("\n# Load the data")
print("import pandas as pd")
print("df = pd.read_csv('output/erfs_rr_for_slopes.csv')")
print("# or")
print("df = pd.read_pickle('output/erfs_rr_for_slopes.pkl')")
print("\n# Filter by country")
print("finland = df[df['country'] == 'Finland']")
print("\n# Filter by city")
print("helsinki = df[df['city'] == 'Helsinki']")
print("\n# Filter by age group")
print("young = df[df['age_group'] == '20-44']")
print("\n# Get specific values")
print("helsinki_young = df[(df['city'] == 'Helsinki') & (df['age_group'] == '20-44')]")
print("rr00 = helsinki_young['RR00'].values[0]")
print(f"\nColumns available: {df_flat.columns.tolist()}")
