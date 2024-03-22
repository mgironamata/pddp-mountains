import numpy as np
import pandas as pd

__all__ = [
              'log_transform',
                'add_year_month_season',
                    'calculate_doy_columns',
                        'add_yesterday_observation',
                            'sort_group_by_column'
        ]

def log_transform(df, columns, epsilon=1e-8):
    """
    Applies a log transformation to specified columns in the DataFrame to handle skewness.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list of str): List of column names to apply the log transformation.
        epsilon (float): Small constant to add to column values to avoid log(0).

    Returns:
        pd.DataFrame: DataFrame with the log-transformed columns.
    """
    for column in columns:
        df[column + '_log'] = np.log(df[column] + epsilon)
    return df

def add_year_month_season(df, date_column='Date'):
    """
    Adds year, month, and season columns to the DataFrame based on a date column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        date_column (str): Name of the column that contains the date information.

    Returns:
        pd.DataFrame: DataFrame with added year, month, and season columns.
    """
    df['year'] = pd.DatetimeIndex(df[date_column]).year
    df['month'] = pd.DatetimeIndex(df[date_column]).month
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['season'] = df['season'].map(seasons)
    return df

def calculate_doy_columns(df, date_column='Date'):
    """
    Adds day of year sine and cosine transformation columns to capture seasonal effects.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        date_column (str): Name of the column that contains the date information.

    Returns:
        pd.DataFrame: DataFrame with added sine and cosine day of year columns.
    """
    doy = pd.DatetimeIndex(df[date_column]).dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return df

def add_yesterday_observation(df, target_column):
    """
    Adds a column to the DataFrame with yesterday's value of the specified target column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target_column (str): The target column to shift and create a 'yesterday' feature.

    Returns:
        pd.DataFrame: DataFrame with added 'yesterday' column for the target feature.
    """
    df[target_column + '_yesterday'] = df[target_column].shift(1)
    return df

def sort_group_by_column(df, group_by_column, sort_by_column):
    """
    Sorts the DataFrame by quantiles within each group defined by `group_by_column`
    based on the values in `sort_by_column`.

    Args:
        df (pd.DataFrame): The DataFrame to be sorted.
        group_by_column (str): The column name to define groups.
        sort_by_column (str): The column name whose quantiles are used for sorting within each group.

    Returns:
        pd.DataFrame: A DataFrame sorted by quantiles within each group.
    """
    # Validate inputs
    if group_by_column not in df.columns or sort_by_column not in df.columns:
        raise ValueError("Specified columns must exist in the DataFrame.")

    # Group by the specified column and sort within each group by the specified sort column
    sorted_df = (df.groupby(group_by_column, group_keys=False)
                   .apply(lambda x: x.sort_values(by=sort_by_column))
                   .reset_index(drop=True))

    return sorted_df

if __name__ == "__main__":

    # Example DataFrame
    data = {
        'Station': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Precipitation': [0.2, 0.1, 0.3, 0.4, 0.2, 0.6],
        'Temperature': [15, 16, 14, 13, 12, 11]
    }
    df = pd.DataFrame(data)

    # Sort by quantile within each station based on Precipitation
    sorted_df = sort_group_by_column(df, group_by_column='Station', sort_by_column='Precipitation')


    print(sorted_df)
