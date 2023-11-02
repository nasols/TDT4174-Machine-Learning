import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

class Data_Manager() : 

    def __init__(self) : 
        # Y_train
        self.train_a = pd.DataFrame() 
        self.train_b = pd.DataFrame()
        self.train_c = pd.DataFrame()

        self.X_train_observed_a = pd.DataFrame()
        self.X_train_observed_b = pd.DataFrame()
        self.X_train_observed_c = pd.DataFrame()

        self.X_train_estimated_a = pd.DataFrame()
        self.X_train_estimated_b = pd.DataFrame()
        self.X_train_estimated_c = pd.DataFrame()

        self.X_test_estimated_a = pd.DataFrame()
        self.X_test_estimated_b = pd.DataFrame()
        self.X_test_estimated_c = pd.DataFrame()

        self.data_A = pd.DataFrame()    
        self.data_B = pd.DataFrame()
        self.data_C = pd.DataFrame()

        # X_train_obs, Y_train_obs
        self.data_A_obs = pd.DataFrame()    
        self.data_B_obs = pd.DataFrame()
        self.data_C_obs = pd.DataFrame()
        
        # X_train_obs, Y_train_obs
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
    
    def dms2dm(self, dms):
        self.train_a = dms.data['train_a']
        self.train_b = dms.data['train_b']
        self.train_c = dms.data['train_c']

        self.X_train_estimated_a = dms.data['X_train_estimated_a']
        self.X_train_estimated_b = dms.data['X_train_estimated_b']
        self.X_train_estimated_c = dms.data['X_train_estimated_c']

        self.X_train_observed_a = dms.data['X_train_observed_a']
        self.X_train_observed_b = dms.data['X_train_observed_b']
        self.X_train_observed_c = dms.data['X_train_observed_c']

        self.X_test_estimated_a = dms.data['X_test_estimated_a']
        self.X_test_estimated_b = dms.data['X_test_estimated_b']
        self.X_test_estimated_c = dms.data['X_test_estimated_c']

        self.data_A_obs = dms.data['data_A_obs']
        self.data_B_obs = dms.data['data_B_obs']
        self.data_C_obs = dms.data['data_C_obs']

        self.data_A_es = dms.data['data_A_es']
        self.data_B_es = dms.data['data_B_es']
        self.data_C_es = dms.data['data_C_es']

        self.data_A = dms.data['data_A']   
        self.data_B = dms.data['data_B']
        self.data_C = dms.data['data_C']

        self.amplitude = dms.data['amplitude']

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
    
    def combine_data(self): 
        """
        Combines datasets A, B and C into one set containing all features and pv_measurements. 

        Combine_observed_estimated (bool) determines if you want one single set for A B C or keep the observed and estimated
        data split 

        Warning! Data should have no NaN values or be of same frequency before combining! 
        """
        weather_data_A = pd.concat([self.X_train_observed_a, self.X_train_estimated_a], axis=0, ignore_index=True)
        weather_data_B = pd.concat([self.X_train_observed_b, self.X_train_estimated_b], axis=0, ignore_index=True)
        weather_data_C = pd.concat([self.X_train_observed_c, self.X_train_estimated_c], axis=0, ignore_index=True)

        self.data_A = pd.merge(weather_data_A, self.train_a, how="left", on="date_forecast")
        self.data_B = pd.merge(weather_data_B, self.train_b,  on="date_forecast", how="left")
        self.data_C = pd.merge(weather_data_C, self.train_c, on="date_forecast", how="left")

        if ( self.data_A.columns.__contains__("date_calc") ): 
            self.data_A = self.data_A.drop("date_calc", axis=1)
            self.data_B = self.data_B.drop("date_calc", axis=1)
            self.data_C = self.data_C.drop("date_calc", axis=1)

        self.data_A = self.data_A.dropna()
        self.data_B = self.data_B.dropna()
        self.data_C = self.data_C.dropna()

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

            #set = set.fillna(0.0)

            imputed_sets.append(set)

        return imputed_sets
    
    def iterative_imputer(self, datasets) :

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer
        from tqdm import tqdm
        imputed_sets = []
        imp = IterativeImputer(random_state=0, missing_values=np.nan, add_indicator=False, imputation_order="ascending", skip_complete=True)

        for set in tqdm(datasets): 

            cols = set.columns 

            if set.columns.__contains__("date_forecast"): 
                dates = set["date_forecast"]
            
            if set.columns.__contains__("date_calc"): 
                set = set.drop("date_calc", axis=1)
            
            cols = set.columns.delete(0)

            set_wo_date = set.drop("date_forecast", axis=1)

            print("getting to imputing")
            #imputing (estimating) missing values 
            imp.fit(set_wo_date)
            
            set_wo_date = pd.DataFrame(imp.transform(set_wo_date), columns=imp.get_feature_names_out())

             # setting column lables basck
            set = set_wo_date
            
            set["date_forecast"] = dates

            # set = set.fillna(0.0)


            #sorting columns 
            cols = cols.tolist()
            cols.insert(0, "date_forecast")

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

            ## set = set.fillna(0.0)

            imputed_sets.append(set)

        return imputed_sets
    
    def normalize_data(self) : 
        from sklearn import preprocessing

        relevant_sets = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and not attr.__contains__("data") and not attr == 'amplitude']
        self.__setattr__("normalizing_consts", {})

        min_max_scaler = preprocessing.MinMaxScaler()
        normalizer = preprocessing.Normalizer()
        

        for att in relevant_sets: 
            set : pd.DataFrame = self.__getattribute__(att)

            cols = set.columns 

            if set.columns.__contains__("date_forecast"): 
                dates = set["date_forecast"]
            
            if set.columns.__contains__("date_calc"): 
                set = set.drop("date_calc", axis=1)

            cols = set.columns.delete(0)

            set_wo_date = set.drop("date_forecast", axis=1)


            x = set_wo_date.values

            x_normalized = min_max_scaler.fit_transform(x)

            
            self.normalizing_consts[att] = (set_wo_date.min(), np.abs(set_wo_date.max() - set_wo_date.min())) ## storing normalizing consts for later 
            
            normalized_set = pd.DataFrame(x_normalized)

            normalized_set.columns = cols


            # setting column lables basck
            
            normalized_set["date_forecast"] = dates

            #sorting columns 
            cols = cols.tolist()
            cols.insert(0, "date_forecast")

            self.__setattr__(att, normalized_set)
 
    def scaling(self, preds, location:str) : 

        """
        FORMAT OF PREDICTIONS SHOULD BE 1 COLUMN WITH PREDS

        LOCATION: A B or C
        """

        relevant_sets = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and not attr.__contains__("data") and not attr == 'amplitude' and (attr.__contains__("train_a") or attr.__contains__("train_b") or attr.__contains__("train_c"))]

        loc_index = 0

        if location.capitalize() == "A" :

            loc_index = 0

        elif location.capitalize() == "B": 
        
            loc_index = 1

        elif location.capitalize() == "C": 

            loc_index = 2


        relevant_set = relevant_sets[loc_index]

    
        min = self.normalizing_consts[relevant_set][0][0]
        diff = self.normalizing_consts[relevant_set][1][0]

        scaled_set = (preds + min) * diff

        return scaled_set
    
    def combine_overlap_BC(self): 
        import math
        """
        This function is created for merging B and C to remove the nan values apparent when merging pv_measurement to the weather data
        This is because of the observation that B and C overlap and where one is missing the other fills in. 
        Must run combine data first to create data_A B C
        """

        original_B = self.data_B
        original_C = self.data_C  

        b2c_scaling = original_B["pv_measurement"].max()/original_C["pv_measurement"].max()

        print(b2c_scaling)      

        original_C[original_C.isnull()] = self.data_B
        original_B[original_B.isnull()] = self.data_C

        self.data_C = original_C.dropna()
        self.data_B = original_B.dropna()

    def sorting_columns_inMainSets(self):

        A = self.data_A 
        cols = A.columns.tolist()

        #sorting columns 
        cols.remove("date_forecast")
        cols.remove("pv_measurement")
        cols.insert(0, "date_forecast")
        cols.insert(0, "pv_measurement")

        A = A[cols]
        self.data_A = A

        #------------------------------------------------------------# 

        B = self.data_B
        cols = B.columns.tolist()

        #sorting columns 
        cols.remove("date_forecast")
        cols.remove("pv_measurement")
        cols.insert(0, "date_forecast")
        cols.insert(0, "pv_measurement")

        B = B[cols]
        self.data_B = B 

        #------------------------------------------------------------#

        C = self.data_C

        cols = C.columns.tolist()

        #sorting columns 
        cols.remove("date_forecast")
        cols.remove("pv_measurement")
        cols.insert(0, "date_forecast")
        cols.insert(0, "pv_measurement")

        C = C[cols]
        self.data_C = C

        

        

        







