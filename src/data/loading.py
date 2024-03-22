import pandas as pd
import logging

def import_dataframe(path, format='csv', verbose=False, **kwargs):
    """
    Reads a DataFrame from a specified file.

    Args:
        path (str): The file path to import.
        format (str, optional): The format of the file ('csv', 'pickle', 'xlsx', etc.). Defaults to 'csv'.
        verbose (bool, optional): If True, print additional information during loading. Defaults to False.
        **kwargs: Arbitrary keyword arguments passed to pandas read function.

    Returns:
        DataFrame: The imported pandas DataFrame.
    """
    try:
        if format == 'csv':
            df = pd.read_csv(path, **kwargs)
        elif format == 'pickle':
            df = pd.read_pickle(path, **kwargs)
        elif format == 'xlsx':
            df = pd.read_excel(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if verbose:
            logging.info(f"Imported dataframe from {path}, shape: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Failed to import dataframe from {path}: {e}")
        raise

