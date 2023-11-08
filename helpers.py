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

def find_const_interval(df, target_attribute, interval_length, ignore_values=[]):
    """
    Find all the intervals of the given length in the dataset where the target_attribute is constant.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    target_attribute (str): The target attribute to check for constant values.
    interval_length (int): The length of the interval to search for.

    Returns:
    list: A list of indexes of all the values in the intervals where the target_attribute is constant for the given interval length or bigger.
    """
    idxs = []
    intervals_found = 0
    i = 0
    while i + interval_length < len(df) - 1:
        if df[target_attribute][i:i+interval_length].nunique() == 1 and df[target_attribute][i] not in ignore_values:
            j = i + interval_length - 1
            value = df[target_attribute][i]
            while value == df[target_attribute][j] and j < len(df) - 1:
                j += 1
            idxs.extend(list(range(i, j)))
            i = j+2
            intervals_found += 1
        else:
            i += 1
    return idxs, intervals_found

def donate_missing_rows(reciever, donor, target_attribute = 'date_forecast', scalepv = False):
    """
    Add rows from donor to reciever where the target_attribute is missing in reciever.

    Parameters:
    reciever (pandas.DataFrame): The dataframe to add rows to.
    donor (pandas.DataFrame): The dataframe to add rows from.
    target_attribute (str): The target attribute to check for missing values.

    Returns:
    tuple: A tuple containing the updated reciever dataframe and the number of rows that were donated.
    """
    donor_copy = donor.copy()
    if scalepv:
        scale = reciever['pv_measurement'].mean() / donor['pv_measurement'].mean()
        donor_copy['pv_measurement'] = donor_copy['pv_measurement'] * scale

    donated_rows = 0
    for date in donor_copy[target_attribute]:
        if date not in reciever[target_attribute].values:
            new_row = donor_copy[donor_copy[target_attribute] == date]
            reciever = pd.concat([reciever, new_row], ignore_index=True)
            donated_rows += 1
    reciever = reciever.sort_values(target_attribute).reset_index(drop=True)

    return reciever, donated_rows



