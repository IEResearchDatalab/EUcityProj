#!/home/daniprec/.conda/envs/masselot/bin/python
"""
Extract RR_for_slopes_results and convert to a clean pandas DataFrame
This script needs "output/erfs.rds" to be present, which is created by the R
script "rscripts/extract_Masselot_data.R" by Simon Lloyd.
"""

import os
from typing import Tuple

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter


def get_paths() -> Tuple[str, str]:
    """Get paths for input RDS file and output directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    rds_path = os.path.join(project_dir, "output", "erfs.rds")
    output_dir = os.path.join(project_dir, "output")
    return rds_path, output_dir


def load_rds_file(rds_path: str):
    """Load RDS file and return the data object."""
    print("Reading RDS file...")
    data = r.readRDS(rds_path)
    return data


def extract_rr_data(data) -> pd.DataFrame:
    """Extract RR_for_slopes_results and convert to flat DataFrame."""
    print("Extracting RR_for_slopes_results...")
    rr_data = data.rx2("RR_for_slopes_results")

    countries = list(r["names"](rr_data))
    print(f"Countries: {len(countries)}")

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

    print("Creating flat DataFrame...")
    df_flat = pd.concat(records, ignore_index=True)
    return df_flat


def print_dataframe_info(df: pd.DataFrame) -> None:
    """Print summary information about the DataFrame."""
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nSummary by country:")
    print(df.groupby("country").size())


def save_dataframe(df: pd.DataFrame, output_dir: str) -> None:
    """Save DataFrame in multiple formats (CSV, Pickle, Parquet)."""
    print("\n" + "=" * 60)
    print("Saving files...")
    print("=" * 60)

    # 1. CSV
    csv_path = os.path.join(output_dir, "erfs_rr_for_slopes.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV: {csv_path}")
    print(f"  Size: {os.path.getsize(csv_path) / 1024:.2f} KB")

    # 2. Pickle
    pickle_path = os.path.join(output_dir, "erfs_rr_for_slopes.pkl")
    df.to_pickle(pickle_path)
    print(f"✓ Pickle: {pickle_path}")
    print(f"  Size: {os.path.getsize(pickle_path) / 1024:.2f} KB")

    # 3. Try Parquet if available
    try:
        parquet_path = os.path.join(output_dir, "erfs_rr_for_slopes.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"✓ Parquet: {parquet_path}")
        print(f"  Size: {os.path.getsize(parquet_path) / 1024:.2f} KB")
    except ImportError:
        print("⚠ Parquet not available (install pyarrow or fastparquet)")


def print_usage_examples(df: pd.DataFrame) -> None:
    """Print usage examples for loading and using the data."""
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
    print(
        "helsinki_young = df[(df['city'] == 'Helsinki') & (df['age_group'] == '20-44')]"
    )
    print("rr00 = helsinki_young['RR00'].values[0]")
    print(f"\nColumns available: {df.columns.tolist()}")


def main() -> None:
    """Main function to orchestrate the conversion process."""
    # Get paths
    rds_path, output_dir = get_paths()

    # Load RDS file
    data = load_rds_file(rds_path)

    # Extract and convert to DataFrame
    df_flat = extract_rr_data(data)

    # Print information
    print_dataframe_info(df_flat)

    # Save in multiple formats
    save_dataframe(df_flat, output_dir)

    # Print usage examples
    print_usage_examples(df_flat)


if __name__ == "__main__":
    main()
