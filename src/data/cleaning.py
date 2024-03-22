import pandas as pd
import numpy as np
import logging

__all__ = ['drop_dataframe_nan_values', 
           'replace_negative_values', 
           'clip_time_period']

def drop_dataframe_nan_values(df, columns=None, inplace=False):
    """
    Drops rows with NaN values from specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        columns (list of str, optional): Columns to check for NaN values. If None, checks all columns. Defaults to None.
        inplace (bool, optional): Whether to drop the rows in place or return a new DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: The cleaned DataFrame, if inplace is False. None otherwise.
    """
    if columns is None:
        columns = df.columns
    try:
        result = df.dropna(subset=columns, inplace=inplace)
        if not inplace:
            return result
    except Exception as e:
        logging.error(f"Error dropping NaN values: {e}")
        raise

def replace_negative_values(df, columns=None, replacement_value=0, inplace=False):
    """
    Replaces negative values in specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        columns (list of str, optional): Columns to check for negative values. If None, checks all columns. Defaults to None.
        replacement_value (int or float, optional): The value with which to replace negative values. Defaults to 0.
        inplace (bool, optional): Whether to replace the values in place or return a new DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: The cleaned DataFrame, if inplace is False. None otherwise.
    """
    if columns is None:
        columns = df.columns
    try:
        if not inplace:
            df = df.copy()
        for column in columns:
            if column in df.columns:
                df[column] = df[column].apply(lambda x: replacement_value if x < 0 else x)
        if not inplace:
            return df
    except Exception as e:
        logging.error(f"Error replacing negative values in {columns}: {e}")
        raise

def clip_time_period(df, start_date, end_date, date_column='Date'):
    """
    Clips a DataFrame to only include rows within a specified time period.

    Args:
        df (pd.DataFrame): The DataFrame to clip.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
        date_column (str, optional): The name of the column containing the date. Defaults to 'Date'.

    Returns:
        pd.DataFrame: The clipped DataFrame.
    """
    try:
        if date_column not in df.columns:
            raise ValueError(f"{date_column} not in DataFrame columns")
        
        mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
        return df.loc[mask]
    except Exception as e:
        logging.error(f"Error clipping time period from {start_date} to {end_date}: {e}")
        raise

if __name__ == "__main__":

    # Sample DataFrame
    df = pd.DataFrame({
        'Date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'],
        'Value': [1, -1, np.nan, 4],
        'Temperature': [-20, -15, -10, np.nan]
    })

    # Cleaning operations
    cleaned_df = drop_dataframe_nan_values(df, columns=['Value'], inplace=False)
    cleaned_df = replace_negative_values(cleaned_df, columns=['Value', 'Temperature'], inplace=False)
    clipped_df = clip_time_period(cleaned_df, '2020-01-02', '2020-01-03')

    print(clipped_df)