import numpy as np
from sklearn.model_selection import train_test_split, KFold

__all__ = ['create_train_test_split', 
           'create_cv_held_out_sets', 
           'time_series_split']

def create_train_test_split(df, test_size=0.2, random_state=None):
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: Returns four DataFrames: X_train, X_test, y_train, y_test
    """
    # This is a placeholder for how you might structure the function. You'd need
    # to adjust it based on how your data is structured and what you're predicting.
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_cv_held_out_sets(df, n_splits=5, shuffle=True, random_state=None):
    """
    Creates cross-validation sets for training and evaluates models.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        n_splits (int): Number of folds. Must be at least 2.
        shuffle (bool): Whether to shuffle the data before splitting into batches.
        random_state (int): When shuffle is True, random_state affects the ordering of the indices.

    Returns:
        generator: A generator that yields train/test indices for each split.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in kf.split(df):
        yield train_index, test_index

# Example of a more specialized function you might need
def time_series_split(df, split_date):
    """
    Splits a time series DataFrame into before and after a specific date.

    Args:
        df (pd.DataFrame): Time series DataFrame to split.
        split_date (str): The date to split the DataFrame on.

    Returns:
        tuple: Two DataFrames, one with data before the split date and one with data after.
    """
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    return train, test

def prepare_data_splits(df, predictors, target, splits_info, sort_func=None):
    """
    Prepares dataset splits for modeling, optionally applying sorting within each split.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.
        predictors (list): List of column names to be used as predictors.
        target (str): Column name to be used as the prediction target.
        splits_info (dict): Dictionary containing split keys and corresponding DataFrame indices or boolean masks.
        sort_func (callable, optional): Function to apply sorting within each split. Must accept a DataFrame and return a DataFrame.

    Returns:
        dict: A dictionary containing prepared data splits ('X' and 'Y' for each split) and the overall mean and standard deviation for predictors.
    """
    data = {}
    x_overall = df[predictors].to_numpy()
    x_mean = x_overall.mean(axis=0)
    x_std = x_overall.std(axis=0)

    for split_key, indices_or_mask in splits_info.items():
        split_df = df.iloc[indices_or_mask] if isinstance(indices_or_mask, np.ndarray) else df[indices_or_mask]
        
        if sort_func:
            split_df = sort_func(split_df)
        
        data[f'X_{split_key}'] = (split_df[predictors].to_numpy() - x_mean) / x_std
        data[f'Y_{split_key}'] = split_df[target].to_numpy()

    return data, x_mean, x_std

