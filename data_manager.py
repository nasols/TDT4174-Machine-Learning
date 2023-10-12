import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

class Data_Manager() : 

    def __init__(self) : 

        self.train_a = pd.DataFrame()
        self.train_b = pd.DataFrame()
        self.train_c = pd.DataFrame()

        self.X_train_estimated_a = pd.DataFrame()
        self.X_train_estimated_b = pd.DataFrame()
        self.X_train_estimated_c = pd.DataFrame()

        self.X_train_observed_a = pd.DataFrame()
        self.X_train_observed_b = pd.DataFrame()
        self.X_train_observed_c = pd.DataFrame()

        self.X_test_estimated_a = pd.DataFrame()
        self.X_test_estimated_b = pd.DataFrame()
        self.X_test_estimated_c = pd.DataFrame()

        self.data_A_obs = pd.DataFrame()
        self.data_B_obs = pd.DataFrame()
        self.data_C_obs = pd.DataFrame()
        
        self.data_A_es = pd.DataFrame()
        self.data_B_es = pd.DataFrame()
        self.data_C_es = pd.DataFrame()

        self.amplitude = np.zeros(3) # amp_a, amp_b, amp_c

    def data_loader(self): 
        """
        Function that loads the datasets into data manager, loads all data 
        """

        self.train_a = pd.read_parquet('A/train_targets.parquet')
        self.train_a = self.train_a.rename(columns={"time":"date_forecast"})

        self.train_b = pd.read_parquet('B/train_targets.parquet')
        self.train_b = self.train_b.rename(columns={"time":"date_forecast"})

        self.train_c = pd.read_parquet('C/train_targets.parquet')
        self.train_c = self.train_c.rename(columns={"time":"date_forecast"})

        self.X_train_estimated_a = pd.read_parquet('A/X_train_estimated.parquet')
        self.X_train_estimated_b = pd.read_parquet('B/X_train_estimated.parquet')
        self.X_train_estimated_c = pd.read_parquet('C/X_train_estimated.parquet')

        self.X_train_observed_a = pd.read_parquet('A/X_train_observed.parquet')
        self.X_train_observed_b = pd.read_parquet('B/X_train_observed.parquet')
        self.X_train_observed_c = pd.read_parquet('C/X_train_observed.parquet')

        self.X_test_estimated_a = pd.read_parquet('A/X_test_estimated.parquet')
        self.X_test_estimated_b = pd.read_parquet('B/X_test_estimated.parquet')
        self.X_test_estimated_c = pd.read_parquet('C/X_test_estimated.parquet')
    
    def drop_feature(datasets:list[pd.DataFrame], features:list[str]):
        """
        Takes in list of datasets and removes features from the sets

        Returns list of altered datasets
        """

        altered_sets = []

        for set in datasets: 
            for feature in features:

                set = set.drop(feature, axis=1)

            altered_sets.append(set)

        return altered_sets
    
    def combine_data(self, combine_observed_estimated = True, fill_nan = False): 
        """
        Combines datasets A, B and C into one set containing all features and pv_measurements. 

        Combine_observed_estimated (bool) determines if you want one single set for A B C or keep the observed and estimated
        data split 

        Warning! Data should have no NaN values or be of same frequency before combining! 
        """

        if (not combine_observed_estimated) : 

            self.data_A_obs = pd.merge(self.train_a, self.X_train_observed_a, on="date_forecast", how="left").dropna()
            self.data_B_obs = pd.merge(self.train_b, self.X_train_observed_b, on="date_forecast", how="left").dropna()
            self.data_C_obs = pd.merge(self.train_c, self.X_train_observed_c, on="date_forecast", how="left").dropna()
            
            self.data_A_es = pd.merge(self.train_a, self.X_train_estimated_a, on="date_forecast", how="left").dropna()
            self.data_B_es = pd.merge(self.train_b, self.X_train_estimated_b, on="date_forecast", how="left").dropna()
            self.data_C_es = pd.merge(self.train_c, self.X_train_estimated_c, on="date_forecast", how="left").dropna()

            return self.data_A_obs, self.data_B_obs, self.data_C_obs, self.data_A_es, self.data_B_es, self.data_C_es

        else : 
            weather_data_A = pd.concat([self.X_train_observed_a, self.X_train_estimated_a], axis=0, ignore_index=True)
            weather_data_B = pd.concat([self.X_train_observed_b, self.X_train_estimated_b], axis=0, ignore_index=True)
            weather_data_C = pd.concat([self.X_train_observed_c, self.X_train_estimated_c], axis=0, ignore_index=True)

            self.data_A = pd.merge(self.train_a, weather_data_A, on="date_forecast")
            self.data_B = pd.merge(self.train_b, weather_data_B, on="date_forecast")
            self.data_C = pd.merge(self.train_c, weather_data_C, on="date_forecast")

            if self.data_A.isna().sum().sum() > 0 :
                warnings.warn("Warning! Data should have no NaN values or be of same frequency before combining! Use impute or interpolation on data before combining! This could also come from dates in the combined datasets not overlapping fully.")

            return self.data_A, self.data_B, self.data_C

    def impute_data(self, datasets, advanced_imputer=False):
        """
        imputes data to fill in missing values

        takes in a list of datasets

        returns list of imputed data

        removes all columns consisting entirely of nan 
        """

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer
        from tqdm import tqdm
        imputed_sets = []

        for set in tqdm(datasets): 

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
            set_wo_date = pd.DataFrame(imp.transform(set_wo_date), columns=imp.get_feature_names_out())
            

            # setting column lables basck
            set = set_wo_date
            
            set["date_forecast"] = dates

            #sorting columns 
            cols = cols.tolist()
            cols.insert(0, "date_forecast")

            set = set.fillna(0.0)

            imputed_sets.append(set)

        return imputed_sets

    def resample_data(self, datasets, freq) : 


        """
        resamples the given dataset into the correct frequency. 
        H : hourly 
        15T : 15min 
        """

        corr = []

        for set in datasets: 
            set_hourly = set.resample(freq, on="date_forecast").mean()

            set_dates = set["date_forecast"].dt.date.unique().tolist()

            set_hourly["date_forecast"] = set_hourly.index

            set_corr = set_hourly[set_hourly["date_forecast"].dt.date.isin(set_dates)]

            set_corr.index = pd.RangeIndex(0, len(set_corr))
            corr.append(set_corr)

        
        return corr

    def add_feature(dataset, feature_name, data) :

        added_set = dataset[feature_name] = data

        return added_set
    
    def set_info(self, dataset:pd.DataFrame):

        (dataset.info())

    def plot_feature(self, dataset:pd.DataFrame, featureName:str):


        
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))

        dataset[['date_forecast', featureName]].set_index("date_forecast").plot(ax=axs, title=featureName, color='red')
       
    def KNNImputing(self, datasets) :
        from sklearn.impute import KNNImputer
        from tqdm import tqdm

        imputed_sets = []

        for set in tqdm(datasets): 

            # storing columns to lable after impute, also removing date column as this does not work with impute 
            cols = set.columns 

            if set.columns.__contains__("date_forecast"): 
                dates = set["date_forecast"]
            
            if set.columns.__contains__("date_calc"): 
                set = set.drop("date_calc", axis=1)

            cols = set.columns.delete(0)

            set_wo_date = set.drop("date_forecast", axis=1)

            #imputing (estimating) missing values 
            imp = KNNImputer(n_neighbors=5)
            imp.fit(set_wo_date)
            set_wo_date = pd.DataFrame(imp.transform(set_wo_date), columns=imp.get_feature_names_out())
            

            # setting column lables basck
            set = set_wo_date
            
            set["date_forecast"] = dates

            #sorting columns 
            cols = cols.tolist()
            cols.insert(0, "date_forecast")

            set = set.fillna(0.0)

            imputed_sets.append(set)

        return imputed_sets
