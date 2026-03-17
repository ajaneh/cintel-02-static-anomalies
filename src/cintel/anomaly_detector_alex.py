# === DECLARE IMPORTS (packages we will use in this project) ===

# First from the Python standard library (no installation needed)
import logging
from pathlib import Path
from typing import Final

import polars as pl
from datafun_toolkit.logger import get_logger, log_header, log_path

# === CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("P2", level="DEBUG")

# === DECLARE GLOBAL CONSTANTS FOR FOLDER PATHS (directories) ===

ROOT_DIR: Final[Path] = Path.cwd()
DATA_DIR: Final[Path] = ROOT_DIR / "data"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"

# === DECLARE GLOBAL CONSTANTS FOR FILE PATHS ===

DATA_FILE: Final[Path] = DATA_DIR / "clinic_data_alex.csv"
OUTPUT_FILE: Final[Path] = ARTIFACTS_DIR / "anomalies_alex.csv"
CHART_FILE: Final[Path] = ARTIFACTS_DIR / "scatter_plot_alex.html"


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Run the pipeline.

    log_header() logs a standard run header.
    log_path() logs repo-relative paths (privacy-safe).
    """
    log_header(LOG, "CINTEL")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    # Log the constants to help with debugging and transparency.
    log_path(LOG, "ROOT_DIR", ROOT_DIR)
    log_path(LOG, "DATA_FILE", DATA_FILE)
    log_path(LOG, "OUTPUT_FILE", OUTPUT_FILE)

    # Call the mkdir() method to ensure it exists
    # The parents=True argument allows it to create any necessary parent directories.
    # The exist_ok=True argument prevents an error if the directory already exists.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    log_path(LOG, "ARTIFACTS_DIR", ARTIFACTS_DIR)

    # ----------------------------------------------------
    # STEP 1: READ CSV DATA FILE INTO A POLARS DATAFRAME (TABLE)
    # ----------------------------------------------------
    # Polars is great for tabular data.
    # We will use the polars package to
    # read csv (comma-separated values) files
    # into a two dimensional table (DataFrame).

    # Call the polars library read_csv() method.
    # Pass in (provide) the DATA_FILE path of the CSV file.
    # Name the result "df" as is customary.
    df: pl.DataFrame = pl.read_csv(DATA_FILE)

    # Visually inspect the file in the data/ folder.
    # It has columns named `age_years` and `height_inches`.
    # The DataFrame height attribute returns the number of rows.
    LOG.info(f"Loaded {df.height} patient records")

    # ----------------------------------------------------
    # STEP 2: DEFINE THRESHOLDS AND DETECT ANOMALIES
    # ----------------------------------------------------
    # An anomaly is any value greater than the threshold we set.
    # Domain rule for this example:
    # Anything above this value is suspicious.
    LOG.info("Assessing Meaningful Thresholds for Anomaly Detection")

    # The second data file has a different distribution of values, so we need to adjust our thresholds accordingly.

    average_values_both_columns: pl.Expr = (
        pl.col("age_years", "height_inches").mean().name.prefix("avg_")
    )
    LOG.info("Calculating average values for age_years and height_inches")
    average_values = df.select(average_values_both_columns)
    LOG.info(f"Average values:\n{average_values}")
    LOG.info("Calculating Z scores for age_years and height_inches")
    z_score = df.select(
        (pl.col("age_years") - pl.col("age_years").mean())
        / pl.col("age_years").std().name.prefix("z_"),
        (pl.col("height_inches") - pl.col("height_inches").mean())
        / pl.col("height_inches").std().name.prefix("z_"),
    )
    LOG.info(f"Z scores:\n{z_score}")

    # Let's plot the data to visually inspect it and help us choose reasonable thresholds for anomalies.
    # We can use the polars plotting capabilities to create a scatter plot of age_years
    # vs height_inches. This will help us see if there are any obvious outliers and where they are located.
    LOG.info("Creating scatter plot of age_years vs height_inches")
    chart = (
        df.plot.point(
            x="age_years",
            y="height_inches",
            color="species",
        )
        .properties(width=500, title="Irises")
        .configure_scale(zero=False)
        .configure_axisX(tickMinStep=1)
    )
    chart.encoding.x.title = "Age (years)"
    chart.encoding.y.title = "Height (inches)"
    # chart.save('chart.html')

    LOG.info(
        "Scatter plot created. Please inspect the plot to identify any obvious outliers and determine reasonable thresholds for anomalies."
    )
    # x is age in years, so 16 is the upper limit for kids
    MAX_REASONABLE_X_VALUE: Final[float] = 16.0

    # y is height in inches, so maybe 6 feet (72 inches) is a reasonable upper limit
    MAX_REASONABLE_Y_VALUE: Final[float] = 72.0

    LOG.info(f"MAX_REASONABLE_X_VALUE: {MAX_REASONABLE_X_VALUE} in years")
    LOG.info(f"MAX_REASONABLE_Y_VALUE: {MAX_REASONABLE_Y_VALUE} in inches")

    # Create a new DataFrame named anomalies_df that contains
    # only the rows where EITHER
    # the age is TOO HIGH OR
    # the height is TOO HIGH.
    # A single pipe (|) is the OR operator in polars.
    # We will use greater than or equal to (>=) to find values at or above the threshold.
    anomalies_df: pl.DataFrame = df.filter(
        (pl.col("age_years") >= MAX_REASONABLE_X_VALUE)
        | (pl.col("height_inches") >= MAX_REASONABLE_Y_VALUE)
    )

    LOG.info(f"Count of anomalies found: {anomalies_df.height}")

    # ----------------------------------------------------
    # STEP 3: SAVE THE OUTPUT ANOMALIES AS EVIDENCE
    # ----------------------------------------------------
    # We call generated files "artifacts".
    # They are important evidence of the work we did and the results we found.
    # We will save the anomalies_df DataFrame as a CSV file in the artifacts/ folder

    # Every Polars DataFrame has a write_csv() method that saves it as a CSV file.
    # Just pass in the full Path to the file you want to create.

    # Lets use the Z values instead of the raw values to find anomalies.

    anomalies_df.write_csv(OUTPUT_FILE)
    LOG.info(f"Wrote anomalies file: {OUTPUT_FILE}")

    LOG.info("========================")
    LOG.info("Pipeline executed successfully!")
    LOG.info("========================")
    LOG.info("END main()")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
