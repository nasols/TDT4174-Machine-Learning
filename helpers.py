import pandas as pd 


def impute (datasets: list[pd.DataFrame]) : 

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    import numpy as np
    import pandas as pd
    

    imputed_sets = []

    for set in datasets: 

        # storing columns to lable after impute, also removing date column as this does not work with impute 
        cols = set.columns 

        if set.columns.__contains__("date_forecast"): 
            dates = set["date_forecast"]
        
        if set.columns.__contains__("date_calc"): 
            set = set.drop("date_calc", axis=1)

        cols = set.columns.delete(0)

        set_wo_date = set.drop("date_forecast", axis=1)

        #imputing (estimating) missing values 
        imp = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=False)
        imp.fit(set_wo_date)
        set_wo_date = imp.transform(set_wo_date)
        set_wo_date = pd.DataFrame(set_wo_date)

        # setting column lables basck
        set = set_wo_date
        set.columns = set.columns.astype(str)
        set.columns = cols
        set["date_forecast"] = dates

        #sorting columns 
        cols = cols.tolist()
        cols.insert(0, "date_forecast")

        set = set[cols]

        set = set.fillna(0.0)

        imputed_sets.append(set)

    return imputed_sets


def drop_feature(datasets:list[pd.DataFrame], features:list[str]):

    altered_sets = []

    for set in datasets: 
        for feature in features:

            set = set.drop(feature, axis=1)

        altered_sets.append(set)

    return altered_sets

def find_const_interval(df, target_attribute, interval_length):
    """
    Find all the intervals of the given length in the dataset where the target_attribute is constant.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    target_attribute (str): The target attribute to check for constant values.
    interval_length (int): The length of the interval to search for.

    Returns:
    list: A list of indexes of all the values in the intervals where the target_attribute is constant for the given interval length or bigger.
    """
    intervals = []
    start = None
    for i in range(len(df)):
        if df[target_attribute][i:i+interval_length].nunique() == 1:
            if start is None:
                start = i
            if i+interval_length == len(df) or df[target_attribute][i+interval_length:].nunique() != 1:
                end = i+interval_length-1
                if end - start + 1 >= interval_length:
                    intervals.extend(list(range(start, end+1)))
                start = None
        else:
            start = None
    return intervals

def find_const_interval(df, target_attribute, interval_length):
    """
    Find all the intervals of the given length in the dataset where the target_attribute is constant.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    target_attribute (str): The target attribute to check for constant values.
    interval_length (int): The length of the interval to search for.

    Returns:
    list: A list of indexes of all the values in the intervals where the target_attribute is constant for the given interval length or bigger.
    """
    intervals = []
    i = 0
    while i < len(df) - 1:
        if df[target_attribute][i:i+interval_length].nunique() == 1:
            j = i - 1
            value = df[target_attribute][i]
            while value == df[target_attribute][j] and j < len(df) - 1:
                j += 1
            intervals.extend(list(range(i, j)))
            i = j+2
        else:
            i += 1
    return intervals


