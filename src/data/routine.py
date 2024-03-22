from loading import import_dataframe
from cleaning import drop_dataframe_nan_values, replace_negative_values, clip_time_period
from splitting import create_train_test_split, create_cv_held_out_sets, time_series_split
from transformations import log_transform, add_cyclic_date_features
from feature_engineering import add_date_features, calculate_doy_columns, add_yesterday_observation, sort_group_by_column

class DataPreprocessing():

    def __init__(self, add_yesterday=False, sort_by_quantile=False):
        self.add_yesterday = add_yesterday
        self.sort_by_quantile = sort_by_quantile

def main():

    # Load the data
    df = import_dataframe()

    # Clean the data
    df = drop_dataframe_nan_values(df)
    df = replace_negative_values(df)
    df = clip_time_period(df, start_date='2010-01-01', end_date='2020-12-31')

    # Add year, month, and season features
    df = add_date_features(df, date_column='Date')

    # Add day of year sine and cosine features
    df = add_cyclic_date_features(df, date_column='Date', drop_date=False)

    # Add yesterday's observation as a feature
    if add_yesterday:
        df = add_yesterday_observation(df, target_column='Value')

    # Filter incomplete years
    # TO DO
    
    # Sort the DataFrame by quantiles within each month
    if sort_by_quantile:
        df = sort_group_by_column(df, group_by_column='month', sort_by_column='Value')

    # FIlter by basin
    # TO DO
        
    return df

def create_station_dict():
    # TO DO
    return None





    
