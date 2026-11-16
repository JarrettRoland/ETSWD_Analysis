import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def flag_extremes(in_series: pd.Series, k: float = 3) -> pd.Series:
    """ This function flags values in the Series where a reading <= 0 or the reading is 
    outside an acceptable range defined by (Q1 - k*IQR, Q3 + k*IQR). With k=3, this is equivalent
    to defining the bounds as Tukey's outer fences (3 times the IQR). The function returns
    a boolean Series. """

    # filter out 0 values and get quantiles
    q1, q3 = in_series[in_series > 0].quantile([0.25, 0.75])

    # lower bound for values
    lower = q1 - k * (q3 - q1)

    # upper bound for values
    upper = q3 + k * (q3 - q1)

    # return a boolean series for filtering
    out_series = (in_series <= 0) | (in_series < lower) | (in_series > upper) | in_series.isna()
    return out_series


def filter_extremes(in_df: pd.DataFrame, columns: list[str], 
                    all_cols = True, k: float = 3) -> pd.DataFrame:
    """ This function drops rows from the input dataframe that are extreme on either any 
    or all of the given columns depending on the all_cols parameter, 
    and returns the filtered dataframe. """

    print("Length of dataframe input to filtering function = ", len(in_df))
    
    # pd.concat with axis = 1 joins each of these series as independent columns
    # axis = 0 will concatenate on rows
    flags = pd.concat([flag_extremes(in_df[col], k) for col in columns], axis=1)

    if all_cols:
        # pd.all() returns a boolean Series. If the axis = 1, the boolean is calculated
        # across an entire row, if axis = 0, boolean is calculated across an entire 
        # column or attribute.
        drop = flags.all(axis=1)
    else:
        drop = flags.any(axis=1)
        
    print(f"Rows extreme on all of {columns}: {drop.sum()}")
    print(f"Rows kept: {(~drop).sum()} of {len(df)}")
    
    out_df = in_df[~drop].reset_index(drop=True)

    print("Length of dataframe after filtering ", len(out_df))
    
    return out_df

def calculate_valid_ranges(in_df: pd.DataFrame, k: float = 14) -> dict[str, tuple[float, float]]:
    """ Calculates the valid range for each numerical column contained within in_df. The valid
    range is defined as the k * IQR. ASSUMES in_df CONTAINS ONLY NUMERIC COLUMNS."""
    
    in_df = in_df[in_df.columns].astype(float)
    valid_ranges = dict()
    for col in in_df.columns:
        
        q1, q3 = in_df.loc[in_df[col] >= 0, col].quantile([0.25, 0.75])
         # lower bound for values
        lower = q1 - k * (q3 - q1)

        # upper bound for values
        upper = q3 + k * (q3 - q1)
        valid_ranges[col] = (lower, upper)
    
    return valid_ranges
    
    
def clean_attributes(in_df: pd.DataFrame,
                     valid_ranges: dict[str, tuple[float, float]] | None = None,
                     lower_q: float = 0.005, upper_q: float = 0.995) -> pd.DataFrame:
    """ This function sets values outside each column's valid range (defined by the valid_ranges dict) to NaN, 
    then clips each attribute to its percentile range and fills gaps with the median. 
    We set the extreme values to NaN first, and then we fill NaNs with the median value because setting 
    the extreme values to the median first would cause those values to influence the median. While the median is the
    least sensitive measure of central tendency, this prevents issues in cases where there are *many* values
    that are known to be out of bounds.
    """

    # convert any integer columns to float, because NaN can't be assigned to integer values
    # numpy uses the CPU's own definition of NaN - which is tied to the CPU data model for 
    # floats!
    out_df = in_df[[col for col in in_df.columns if col in valid_ranges.keys()]].astype(float)

    # valid_ranges is a dictionary of {"attribute": (lower_bound, upper_bound)}
    if len(valid_ranges) > 0:
        for col, (low, high) in valid_ranges.items():
            if col in out_df:
                # if the acceptable values for the attribute in question are out of bounds
                # assign NaN
                out_filter = (out_df[col] < low) | (out_df[col] > high)
                out_df.loc[out_filter, col] = np.nan
                

    # clip each column to the bounds defined by lower_q and upper_q
    for col in valid_ranges.keys():
        lower, upper = out_df[col].quantile([lower_q, upper_q])
        out_df[col] = out_df[col].clip(lower, upper)


    nan_count_series = out_df.isna().sum(axis=1)
    count=0
    for row in nan_count_series:
        if row > 0:
            count = count + 1
    nans_count = count
    out_df = out_df.fillna(out_df.median())
    print(f"Total NaN values replaced with the Median: {nans_count}")
    print(f"Total out_df rows: {len(out_df)}")
    return out_df


def run_pca(in_df: pd.DataFrame) -> dict[str, Any]:
    """This function standardizes the attributes of the input dataframe and calculates PCA on the columns.
       A dictionary containing the fitted model, a dataframe of eigenvalues, explained variance, and 
       cumulative variance per principle component, a dataframe of how each attribute of in_df correlates
       to each principle component, and the PCA scores for each value of each attribute. 
       
       PCA scores are values for how much each row influences each individual principle component. 
       The individual principle component score for each row is determined by the z-score of the attribute
       multiplied by the weight (component) of that PC.
       
       NOTE: PCA is sensitive to absolute magnitude differences within data, so it is important to normalize
       the dataframe before passing it to run_pca()!"""

    pca = PCA()
    scores = pca.fit_transform(in_df)
    pcs = [f"PC{i + 1}" for i in range(in_df.shape[1])]

    variance_df = pd.DataFrame({
        "Eigenvalue": pca.explained_variance_,
        "Explained Variance": pca.explained_variance_ratio_,
        "Cumulative Variance": np.cumsum(pca.explained_variance_ratio_),
    }, index=pcs)

    # Loadings are the correlation of each attribute with each component
    loadings_df = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                               index=in_df.columns, columns=pcs)
    scores_df = pd.DataFrame(scores, columns=pcs)

    # input dataframe with scores attached for each reading
    out_df = pd.concat([in_df, scores_df], axis=1)

    return {"pca": pca, "variance": variance_df, "loadings": loadings_df, "scores": scores_df, "output": out_df}


def datamine_pca(in_df: pd.DataFrame, pca_vars: list[str],
                 k_start: int, k_stop: int, out_dir: str) -> None:
    """ Runs the clean -> scale -> PCA pipeline for each k from k_start to k_stop (inclusive)
    and writes "PCA Variances K={k}.xlsx" and "PCA Loadings K={k}.xlsx" to out_dir."""
 
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    for k in range(k_start, k_stop + 1):
        valid_ranges = calculate_valid_ranges(in_df[pca_vars], k)
        cleaned_df = clean_attributes(in_df, valid_ranges)
    
    
        scaled_df = pd.DataFrame(StandardScaler().fit_transform(cleaned_df[pca_vars]),
                                 columns=pca_vars, index=cleaned_df.index)
    
        pca_dict = run_pca(scaled_df)
        pca_dict["variance"].to_excel(out_dir / f"PCA Variances K={k}.xlsx")
        pca_dict["loadings"].to_excel(out_dir / f"PCA Loadings K={k}.xlsx")
 
# datamine_pca(main_df, pca_vars, k_start=1, k_stop=20,
#       out_dir=r"C:\path\to\Workover Analysis\PCA Mining")  # <-- your directory here
 




    
    
def supplementary_corr(in_df: pd.DataFrame, scores_df: pd.DataFrame,
                       corr_cols: list[str], num_pcs: int = 4) -> pd.DataFrame:
    """ This function correlates columns that were not used in PCA to the created principal components. 
    In_df and scores_df should be of the same length. num_pcs represents the number of principal components
    in the scores_df to compare against.
    """

    # Assumes that scores_df is organized from highest loading to lowest loading.
    pcs = scores_df.columns[:n_pcs]
    corr_df = pd.DataFrame({pc: [in_df[col].corr(scores_df[pc]) for col in corr_cols] for pc in pcs},
                           index=corr_cols)

    return corr_df


def plot_scree(variance_df: pd.DataFrame, input_path: str | Path | os.PathLike) -> None:
    """ Create scree plot to identify the number of principal components to use 
    in PCA. """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variance_df.index, variance_df["Explained Variance"], label="Per Component")
    ax.plot(variance_df.index, variance_df["Cumulative Variance"], color="tab:orange", marker="o", label="Cumulative")
    ax.set_ylabel("Share of Variance")
    ax.set_title("Scree Plot")
    ax.legend()
    plt.xticks(rotation=45)
    fig.tight_layout()
    out_path = Path(input_path)
    fig.savefig(out_path / "PCA Scree Plot.png", dpi=150)


def plot_biplot(pca_results: dict[str, Any], groups: pd.Series, input_path: str | Path | os.PathLike) -> None:
    """ Plots PC1 vs PC2 scores with the attribute loadings as arrows.
    Rows in a group other than "None" (e.g. workover months) are highlighted. """
    scores_df = pca_results["scores"]
    loadings_df = pca_results["loadings"]
    variance_df = pca_results["variance"]

    fig, ax = plt.subplots(figsize=(7, 6))
    is_none = (groups == "None").to_numpy(dtype=bool)
    ax.scatter(scores_df.loc[is_none, "PC1"], scores_df.loc[is_none, "PC2"],
               s=4, alpha=0.25, color="gray", label="Well-Month")
    for group in groups[~is_none].unique():
        mask = (groups == group).values
        ax.scatter(scores_df.loc[mask, "PC1"], scores_df.loc[mask, "PC2"], s=30, label=group)

    scale = scores_df[["PC1", "PC2"]].abs().max().max() / loadings_df[["PC1", "PC2"]].abs().max().max() * 0.6
    for attribute, row in loadings_df.iterrows():
        ax.arrow(0, 0, row["PC1"] * scale, row["PC2"] * scale, color="black", width=0.01, head_width=0.12)
        ax.text(row["PC1"] * scale * 1.1, row["PC2"] * scale * 1.1, attribute, fontsize=8)

    ax.set_xlabel(f"PC1 ({variance_df.loc['PC1', 'Explained Variance']:.0%})")
    ax.set_ylabel(f"PC2 ({variance_df.loc['PC2', 'Explained Variance']:.0%})")
    ax.set_title("PCA Biplot")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = Path(input_path)
    fig.savefig(out_path / "PCA Biplot.png", dpi=150)

