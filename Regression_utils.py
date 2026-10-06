import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

""" This module will house utility functions that need to be created for 
simplifying views of data models and results. It is assumed that all parameters passed
will either be numpy arrays, pandas Series, or pandas DataFrames. """


def regressionSummary(y_actual:pd.Series | np.array, y_predicted:pd.Series | np.array, x_df:pd.DataFrame):
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

def R_squared(y_actual:pd.Series | np.array, y_predicted:pd.Series | np.array):
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
    file_path: str | os.PathLike[str] | None = None,
    model_type: str | None = "gls"
) -> sm.regression.linear_model.OLS | sm.regression.linear_model.GLS:

    # prevent [const] from showing up in the names list as if it were 
    # a variable
    exog_names = exog_vars.columns.tolist()
    # check that the Series being passed in are of the same length - pass for now
    exog_vars = sm.add_constant(exog_vars)

    # what we are comparing/regressing
    summary_stmt = f"{endog_vars.name} explained by {list(exog_names)} via linear regression"
    if model_type == "ols":
        lm = sm.OLS(endog_vars, exog_vars).fit()
    elif model_type == "gls":
        lm = sm.GLS(endog_vars, exog_vars).fit()
        
    if file_path is None:
        print(summary_stmt, lm.summary())
    else:
        file_path = Path(file_path)
        with open(file_path, "a") as fhand:
            print(summary_stmt, lm.summary(), file=fhand)
    return lm

def plot_ls(
    endog_vars: pd.DataFrame | pd.Series,
    exog_vars: pd.DataFrame | pd.Series, 
    linear_model: sm.regression.linear_model.OLS | sm.regression.linear_model.GLS,
    folder_path: str | os.PathLike[str] | None = None 
) -> None:

    # plot observed data
    fig, ax = plt.subplots()

    # we can't plot more than two dimensions
    num_endogs = _get_number_columns(endog_vars)
    num_exogs = _get_number_columns(exog_vars)
    if num_endogs and num_exogs == 1:
        ax.scatter(endog_vars, exog_vars, label="Data")

        if folder_path is not None:
            file_path = f"{endog_vars.name} vs {exog_vars.columns} OLS.pdf"
            file_path = Path(folder) / file_path
            fig.savefig(file_path, bbox_inches="tight")

    # plot predicted data
    fig, ax = plt.subplots()
    if num_endogs == 1:
        ax.plot(endog_vars, linear_model.fittedvalues, linestyle="--", label="Predicted")
        
        if folder_path is not None:
            file_path = f"{endog_vars.name} vs predicted values.pdf"
            file_path = Path(folder) / file_path
            fig.savefig(file_path, bbox_inches="tight")
        
    return None

def _get_number_columns(in_data: pd.DataFrame | pd.Series) -> int:
    """ Returns number of attributes for DataFrame or Series. Helper function
    to plot_ls(). """
    # if in_data is a dataframe - ndim = 2, if series - ndim = 1
    if in_data.ndim > 1:
        # df.shape returns a tuple (#rows, #columns) we just want columns
        return in_data.shape[1]
    else:
        return 1

def _get_attrs_from_dataframe(in_vars: pd.DataFrame | pd.Series):
    """ Returns either a list of strings representing
    the columns of a DataFrame or Series. """
    if isinstance(in_vars, pd.DataFrame):
        # df.columns returns a pandas Index, which implements the __iter__() function
        # method, so that list(df.columns) calls that iter function and packages it in a list
        return list(in_vars.columns)
    elif isinstance(return list(in_vars.name))
        # calling list(in_vars.name) will return a list of characters rather
        # than a single string as a list
        return [in_vars.name]
    
def datamine_LS(endog_vars: pd.DataFrame | pd.Series,
                exog_vars: pd.DataFrame | pd.Series,
                in_df: pd.DataFrame,
                output_path: str | os.PathLike[str],
                input_path: str | os.PathLike[str] = os.getcwd(),
                ) -> None:
    """ Run a least squares type linear regression for every possible combination of provided exogeneous and endogeneous
    variables. Supports OLS and GLS currently. Writes to a text file at output path. """ 
    
    # set current dir path to input path
    folder = os.chdir(input_path)
    folder = Path(folder)
    out_file = "LS_summaries.txt"
    out_path = folder/file

    # compare every possible singular endogeneous variable against all
    # possible combinations of exogeneous variables.
    for endog in endogs:
        # prevent endog == exog
        possible_exogs = [x for x in exogs if x != endog]
        for r in range(1, len(possible_exogs) + 1):
            # combinations returns a tuple, needs to be cast to list
            for comb in combinations(possible_exogs, r):
                # we'll need to do some filterings on our exog where there are 0 values.
                filt = (in_df[list(vars)] > 0).all(axis=1)
                filt_data = in_df.loc[filt, [endog, *comb]]
                lm = get_ls(
                    filt_data[endog], 
                    filt_data[list(vars)], 
                    filt_data,
                    file_path = out_path
                 )
        