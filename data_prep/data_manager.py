import os
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


    def data_loader(self): 

        """
        Function that loads the datasets into data manager, loads all data 
        """

        self.train_a = pd.read_parquet('A/train_targets.parquet')
        self.train_b = pd.read_parquet('B/train_targets.parquet')
        self.train_c = pd.read_parquet('C/train_targets.parquet')

        self.X_train_estimated_a = pd.read_parquet('A/X_train_estimated.parquet')
        self.X_train_estimated_b = pd.read_parquet('B/X_train_estimated.parquet')
        self.X_train_estimated_c = pd.read_parquet('C/X_train_estimated.parquet')

        self.X_train_observed_a = pd.read_parquet('A/X_train_observed.parquet')
        self.X_train_observed_b = pd.read_parquet('B/X_train_observed.parquet')
        self.X_train_observed_c = pd.read_parquet('C/X_train_observed.parquet')

        self.X_test_estimated_a = pd.read_parquet('A/X_test_estimated.parquet')
        self.X_test_estimated_b = pd.read_parquet('B/X_test_estimated.parquet')
        self.X_test_estimated_c = pd.read_parquet('C/X_test_estimated.parquet')


        self.data_A = pd.DataFrame()


dm = Data_Manager()

dm.data_loader()

print("a: ", dm.data_A)