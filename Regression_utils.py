import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import RegressionResultsWrapper


""" This module will house utility functions that need to be created for 
simplifying views of data models and results. It is assumed that all parameters passed
will either be numpy arrays, pandas Series, or pandas DataFrames. """


def regressionSummary(y_actual:pd.Series | np.ndarray, 
                      y_predicted:pd.Series | np.ndarray, 
                      x_df:pd.DataFrame):
    """ Returns regression characteristics from comparison of
    actual (historical) values to estimated (predicted) ones. """

    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)
    y_residual = y_actual - y_predicted

    metrics = [
        ('Mean Error (ME): ', sum(y_residual) / len(y_residual)),
        ('Root Mean Squared Error (RMSE): ', math.sqrt(mean_squared_error(y_actual, y_predicted))),
        ('Mean Absolute Error (MAE): ', sum(abs(y_residual)) / len(y_residual)),
        ('Mean Percent Error (MPE): ', 100 * sum(y_residual / y_actual) / len(y_residual)),
        ('Mean Absolute Percent Error (MAPE): ', 100 * sum(abs(y_residual / y_actual) / len(y_residual))),
        ('R squared : ', R_squared(y_actual, y_predicted)),
        ('Adjusted R squared: ', adj_R_squared(y_actual, y_predicted, x_df))
    ]

    for metric in metrics:
        print(metric)
    print("\n\n")

    
def R_squared(y_actual:pd.Series | np.ndarray, y_predicted:pd.Series | np.ndarray):
    """ Calculates R squared, which is equivalent to 1 - (sum of squared residuals/sum of squared total).
    y_actual and y_predicted are numpy arrays, or pandas series, or a dataframe slice.

    A residual is equal to the actual value minus the predicted value.
    An error, for sum of square errors, is equal to the actual value minus the average """

    # residual is (y_actual - y_predicted), ss_residual is the squared residual
    ss_residual = np.sum((y_actual - y_predicted) ** 2)

    # error is (y_actual - np.mean(y_actual)), ss_total is the squared error
    ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r_square = 1 - (ss_residual / ss_total)
    return r_square


def adj_R_squared(y_actual, y_predicted, x_df):
    """ Calculates the adjusted R squared value using the R_squared function.
     Adjusted R squared is equal to 1 - ( (sum of squared residuals/degrees freedom residual) / (sum of squared total/degrees freedom total) )
     where degrees freedom of the residual = n - p (#vars) - 1 and degrees freedom total = n - 1

     This gives us adj_R_squared = 1 - ( ((1 - R^2) * (n - 1)) / n - p - 1) where n = num observations and
                                                                                  p = num variables
                                                                                  
     Make sure that x_df only includes the parameters that you are concerned with so that it doesn't count every column
     as a parameter! """
    r_squared = R_squared(y_actual, y_predicted)
    n = len(y_actual)
    p = x_df.shape[1]

    adj_r = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    return adj_r


# function for running OLS on an arbitrary number of variables
# returns the linear model whose output it summarizes
# file path is used for writing the results to file to prevent from overwhelming the visual display
def get_ls(
    endog_vars: pd.DataFrame | pd.Series, 
    exog_vars: pd.DataFrame | pd.Series, 
    model_type: str | None = "gls"
) -> RegressionResultsWrapper:
    """ Returns a fitted linear model (OLS or GLS) based on the regression of
    endog_vars by exog_vars. """
    
    # check that the Series being passed in are of the same length - pass for now
    exog_vars = sm.add_constant(exog_vars)

    if model_type == "ols":
        lm = sm.OLS(endog_vars, exog_vars)
        
    elif model_type == "gls":
        lm = sm.GLS(endog_vars, exog_vars)

    else:
        raise ValueError(f"Unsupported model type provided {model_type}. Use 'ols' or 'gls'")
        
    return lm.fit()

def _save_or_show(fig: plt.Figure,
                  folder_path: str | os.PathLike[str] | None,
                  file_name: str,
                  ) -> None:
    """ Save matplotlib fig to folder_path/file_name and close it, 
        or show it if no folder is given. """
    if folder_path is None:
        plt.show()
    else:
        fig.savefig(Path(folder_path) / file_name, bbox_inches="tight")
        # close so hundreds of models don't pile up open figures in memory
        plt.close(fig)


def plot_ls(linear_model: RegressionResultsWrapper,
            folder_path: str | os.PathLike[str] | None = None,
            ) -> None:
    """ Plot a fitted model from datamine_LS(). The data comes from the model
    itself, so it always matches the filtered rows the model was fit on.
 
    Always plots observed vs predicted values, which works for any number of
    exogs. With a single exog, also plots the data with the fitted line.
    Saves each plot as a PDF in folder_path if given, otherwise shows them. """
    model = linear_model.model
    y = np.asarray(model.endog)
    fitted = np.asarray(linear_model.fittedvalues)
    y_name = model.endog_names
    x_names = [name for name in model.exog_names if name != "const"]
    title = f"{y_name} vs {', '.join(x_names)} ({type(model).__name__})"
 
    if folder_path is not None:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
 
    # observed vs predicted: works for any number of exogs
    fig, ax = plt.subplots()
    ax.scatter(fitted, y, label="Data")
    lims = [min(fitted.min(), y.min()), max(fitted.max(), y.max())]
    ax.plot(lims, lims, linestyle="--", color="black", label="Perfect fit")
    ax.set(xlabel=f"Predicted {y_name}", ylabel=f"Observed {y_name}", title=title)
    ax.legend()
    _save_or_show(fig, folder_path, f"{title} predicted.pdf")
 
    # data with fitted line: only possible in two dimensions
    if len(x_names) == 1:
        x = np.asarray(model.exog)[:, model.exog_names.index(x_names[0])]
        order = np.argsort(x)  # sort so the line is drawn left to right
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Data")
        ax.plot(x[order], fitted[order], linestyle="--", color="black", label="Fitted")
        ax.set(xlabel=x_names[0], ylabel=y_name, title=title)
        ax.legend()
        _save_or_show(fig, folder_path, f"{title} fit.pdf")
    plot_residuals(linear_model, folder_path)
 
 
def plot_residuals(linear_model,
                   folder_path: str | os.PathLike[str] | None = None,
                   ) -> None:
    """ Plot a model's residuals to check normality: a histogram with a normal
    curve overlaid, and a Q-Q plot. Saves a PDF to folder_path if
    given, otherwise shows the plot. """
    model = linear_model.model
    resid = np.asarray(linear_model.resid)
    x_names = [name for name in model.exog_names if name != "const"]
    title = f"{model.endog_names} vs {', '.join(x_names)} ({type(model).__name__})"
    omnibus_p = stats.normaltest(resid).pvalue
 
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
 
    # histogram vs. the normal curve with the residuals' own mean and std
    ax_hist.hist(resid, bins="auto", density=True, alpha=0.7, label="Residuals")
    grid = np.linspace(resid.min(), resid.max(), 200)
    ax_hist.plot(grid, stats.norm.pdf(grid, resid.mean(), resid.std()),
                 color="black", linestyle="--", label="Normal")
    ax_hist.set(xlabel="Residual", ylabel="Density", title="Histogram")
    ax_hist.legend()
 
    # Q-Q plot: points fall on the line if the residuals are normal
    sm.qqplot(resid, line="s", ax=ax_qq)
    ax_qq.set_title("Q-Q plot")
    ax_qq.set_xlabel("Ideal")
    ax_qq.set_ylabel("Sample")
 
    fig.suptitle(f"{title} residuals | Omnibus p = {omnibus_p:.3f}")
    _save_or_show(fig, folder_path, f"{title} residuals.pdf")
 

def _get_attr_names(in_vars: pd.DataFrame | pd.Series | Iterable[str] | str) -> list[str]:
    """ Turn a DataFrame / Series / iterable of column names into a list of column names. """
    if isinstance(in_vars, pd.DataFrame):
        return in_vars.columns.tolist()
    elif isinstance(in_vars, pd.Series):
        return [in_vars.name]
    elif isinstance(in_vars, str):
        return [in_vars]
    return list(in_vars)


def _get_number_columns(in_data: pd.DataFrame | pd.Series) -> int:
    """ Returns number of attributes for DataFrame or Series. Helper function
    to plot_ls(). """
    # if in_data is a dataframe - ndim = 2, if series - ndim = 1
    if in_data.ndim > 1:
        # df.shape returns a tuple (#rows, #columns) we just want columns
        return in_data.shape[1]
    else:
        return 1


def datamine_ls(endog_vars: pd.DataFrame | pd.Series | Iterable[str] | str,
                exog_vars: pd.DataFrame | pd.Series | Iterable[str] | str,
                in_df: pd.DataFrame,
                ) -> dict:
    """ Run a least squares type linear regression for every possible combination
    of provided exogeneous and endogeneous variables.
    Supports OLS and GLS currently. Returns a dict of fitted models keyed by
    "<endog> explained by <exogs>". """
    lm_dict = dict()
 
    # accept DataFrames, Series, or plain column names
    endogs = _get_attr_names(endog_vars)
    exogs = _get_attr_names(exog_vars)
 
    # compare every possible singular endogeneous variable against all
    # possible combinations of exogeneous variables.
    for endog in endogs:
        # prevent endog == exog
        possible_exogs = [x for x in exogs if x != endog]
        for r in range(1, len(possible_exogs) + 1):
            for comb in combinations(possible_exogs, r):
                # combinations returns a tuple, needs to be cast to list
                exog_cols = list(comb)
                # we'll need to do some filterings on our exog where there are 0 values.
                filt = (in_df[exog_cols] > 0).all(axis=1)
                filt_data = in_df.loc[filt, [endog, *exog_cols]].dropna()
                lm = get_ls(
                    filt_data[endog],
                    filt_data[exog_cols],
                )
                lm_dict[f"{endog} explained by {', '.join(exog_cols)}"] = lm
 
    return lm_dict
 

def write_ls_summaries(lm_dict: dict,
                       input_path: str | os.PathLike[str] | None=None ,
                       ) -> None:
    """ Sort the models from datamine_ls() and write their summaries to
    LS_summaries.txt at input_path. Models are grouped by endog; within each
    group, highest adjusted R² comes first, with lowest AIC breaking ties. """
    if input_path is None:
        # Python parameters are evaluated at definition time,
        # this line ensures that getcwd() is the current dir
        # for the caller and not the function itself.
        input_path = os.getcwd()
    out_dir = Path(input_path)

    # create a dictionary of linear models keyed by the endogeneous variables
    # defaultdict() allows a default value to be created when a key is accessed
    # to prevent Key Errors 
    endog_group = defaultdict(list)
    for label, lm in lm_dict.items():
        endog_group[lm.model.endog_names].append((label, lm))
        
    for endog, items in endog_group.items():
     
        # Python's sort is stable: items that tie on the current key keep the
        # order from the previous sort. So sort by the least important key first
        # and the most important key last.
        items.sort(key=lambda item: item[1].aic) # tie-breaker: lowest AIC
        items.sort(key=lambda item: item[1].rsquared_adj, reverse=True)   # highest adj R²

        # Remove any non-unicode characters, periods, or hyphens that might not
        # be filename safe.
        safe_name = re.sub(r"[^\w.-]+", "_", str(endog)).strip("_") or "endog"
        out_path = out_dir / f"LS_summaries_{safe_name}.txt"
    
        with out_path.open("w", encoding="utf-8") as f:
            for label, lm in items:
                f.write(f"{label} | adj R² = {lm.rsquared_adj:.4f} | AIC = {lm.aic:.2f}\n")
                f.write(lm.summary().as_text() + "\n\n")  