import os

import pandas as pd
import requests
from matplotlib.pylab import double


def download_eurostat_data(dataset: str) -> pd.DataFrame:
    """
    Download Eurostat data from the given dataset URL.

    Parameters
    ----------
    dataset : str
        The dataset name to download from Eurostat.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the downloaded data.
    """
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        + dataset
        + "?format=TSV&compressed=true"
    )
    # Create a cache directory if it doesn't exist
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    path_file = os.path.join(cache_dir, "dataset.csv.gz")
    response = requests.get(url)
    if response.status_code == 200:
        with open(path_file, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download file: {response.status_code}")

    df = pd.read_csv(
        path_file,
        compression="gzip",
        encoding="utf-8",
        sep=",|\t",
        na_values=":",
        engine="python",
        dtype_backend="pyarrow",
    )

    # If a column name has "\", drop all after the first "\" in that column name
    df.columns = df.columns.str.split("\\").str[0]

    # The columns which name starts with any year "YYYY" are all numeric
    # We force that numeric columns to be float64
    for col in df.columns:
        # Check the column name matches our criteria
        if col.startswith(tuple(str(year) for year in range(1900, 2100))):
            # Convert the column to numeric, forcing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Some columns have trailing spaces, we remove them
        df.rename(columns={col: col.rstrip()}, inplace=True)

    # Remove the gzip file after reading
    try:
        os.remove(path_file)
    except PermissionError:
        print(
            f"Warning: Could not delete temporary file {path_file}. You may need to delete it manually."
        )
    except OSError as e:
        print(
            f"Warning: Error while trying to delete temporary file {path_file}: {str(e)}"
        )

    # Print date range
    date_columns = df.dropna(axis=0, how="any").columns[
        df.columns.str.match(r"^\d{4}$")
    ]
    if not date_columns.empty:
        start_year = date_columns.min()
        end_year = date_columns.max()
        print(f"[INFO] Eurostat - Date range: {start_year} - {end_year}")

    # Attempt to infer better dtypes for object columns
    return df.infer_objects(copy=False)


def main(path_metadata="./data/metadata.csv"):
    # Load metadata
    metadata = pd.read_csv(path_metadata)

    # Filter the columns we want
    # URAU_CODE: Unique city code
    # URAU_NAME: City name
    # CNTR_CODE: Country code
    # pop: Population in that city (2019)
    columns = ["URAU_CODE", "URAU_NAME", "CNTR_CODE", "pop"]
    df = metadata[columns].drop_duplicates()

    # We know some cities overlap many countries, but for simplicity we will
    # assume they belong to the country they are associated with in the metadata
    # Add the total population per country
    country_pop = df.groupby("CNTR_CODE")["pop"].sum().reset_index()
    # Include country population in the dataframe
    df = df.merge(country_pop, on="CNTR_CODE", suffixes=("", "_studied"))
    # Turn pop columns to int
    df["pop"] = df["pop"].astype(int)
    df["pop_studied"] = df["pop_studied"].astype(int)

    # Get the total population in the country from Eurostats
    df_pop = download_eurostat_data(dataset="demo_pjan")
    # Get both 2018 and 2019. Required because there is
    # no Italy data for 2019, and no France data for 2018.
    # dtype: double[pyarrow] for columns 2018 and 2019
    # We need to change the dtype to int64 to perform calculations
    df_pop["2018"] = df_pop["2018"].astype(double)
    df_pop["2019"] = df_pop["2019"].astype(double)
    df_pop = df_pop.groupby("geo")[["2018", "2019"]].sum().reset_index()
    # Keep only the highest between 2018 and 2019
    df_pop["pop_country"] = df_pop[["2018", "2019"]].max(axis=1).astype(int)
    df_pop.rename(columns={"geo": "CNTR_CODE"}, inplace=True)
    # Keep only necessary columns
    df_pop = df_pop[["CNTR_CODE", "pop_country"]]
    # Merge with the main dataframe
    df = df.merge(df_pop, on="CNTR_CODE", how="left")
    # Calculate the percentage of population studied in each country
    df["pop_percentage"] = (df["pop_studied"] / df["pop_country"]) * 100
    # Save to CSV
    df.to_csv("city_list_with_population.csv", index=False)


if __name__ == "__main__":
    main()
