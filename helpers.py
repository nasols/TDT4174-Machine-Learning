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





