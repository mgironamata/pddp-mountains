import pandas as pd

def FilterByList(df,series,value_list):
    return df[df[series].isin(value_list)]

def filter_by_series(df, series, value):
    return df[df[series]==value]

def list_groups_by_difference(df, group_by_column, field_a, field_b, diff_threshold=0):
    """
    Lists groups where the difference between two specified fields meets a certain threshold.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        group_by_column (str): The column name to define groups.
        field_a (str): The first field for comparison.
        field_b (str): The second field for comparison.
        diff_threshold (float, optional): The threshold for the difference between the two fields to consider. Defaults to 0.

    Returns:
        np.array: An array of unique groups where the difference between field_a and field_b meets the threshold.
    """
    # Calculate the difference between the two specified fields
    df['temp_diff'] = df[field_a] - df[field_b]
    
    # Group by the specified column and calculate the sum of differences within each group
    grouped_diff = df.groupby(group_by_column)['temp_diff'].sum().reset_index()
    
    # Identify groups where the absolute sum of differences meets or exceeds the threshold
    groups_meeting_criteria = grouped_diff[group_by_column][grouped_diff['temp_diff'].abs() >= diff_threshold].unique()
    
    # Clean up the temporary column
    df.drop(columns=['temp_diff'], inplace=True)
    
    return groups_meeting_criteria

import numpy as np

def simulate_binomial_outcomes(df, condition_column, true_condition, probability=None, outcome_column='simulated_outcome', verbose=False):
    """
    Simulates binomial outcomes based on a specified condition within a DataFrame and appends the results as a new series.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        condition_column (str): The column name to apply the condition check.
        true_condition (callable or value): The condition that defines a "success" in binomial terms. Can be a callable (function) that takes a series and returns a boolean series, or a specific value for equality check.
        probability (float, optional): The probability of success for each trial. If None, it's calculated based on the condition. Must be a value in the interval [0,1].
        outcome_column (str, optional): The name of the column to store simulated outcomes. Defaults to 'simulated_outcome'.
        verbose (bool, optional): If True, print additional information. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the simulated binomial outcome column appended.
    """
    # Determine the condition matches
    if callable(true_condition):
        condition_matches = true_condition(df[condition_column])
    else:
        condition_matches = df[condition_column] == true_condition
    
    # Calculate the probability if not provided
    if probability is None:
        probability = condition_matches.mean()
    
    # Validate probability
    if not (0 <= probability <= 1):
        raise ValueError("Probability must be between 0 and 1.")
    
    # Simulate binomial outcomes
    df[outcome_column] = np.random.binomial(n=1, p=probability, size=len(df))
    
    if verbose:
        print(f"Simulated outcomes added to '{outcome_column}' with success probability: {probability}")
    
    return df

def disjunctive_union_lists(li1, li2):
    """ Returns the disjunctive union or symmetric difference between to sets, expressed as lists.
    
    Args:
        li1 (list): first set.
        li2 (list): second set.
    
    Returns:
        list
    """
    
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

if __name__ == "__main__":

    # Sample DataFrame
    df = pd.DataFrame({
        'Group': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Value_A': [1, 2, 3, 4, 5, 6],
        'Value_B': [2, 2, 3, 3, 6, 6]
    })

    # Simulate binomial outcomes
    df = simulate_binomial_outcomes(df, condition_column='Group', true_condition='A', probability=0.8, verbose=True)
    print(df)