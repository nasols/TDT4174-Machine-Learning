from keras import Sequential
import pandas as pd 
import numpy as np 
from keras.layers import LSTM, Dense, GRU, Dropout 

class RNN_Network(): 

    def __init__(self) -> None:
        self.model_A:Sequential = None 
        self.model_B:Sequential = None 
        self.model_C:Sequential = None 

        self.train_A:np.array = None
        self.train_B:np.array = None
        self.train_C:np.array = None

    def create_dataset_timeseries(self, X, y):
        from sklearn.model_selection import TimeSeriesSplit

        ts_cv = TimeSeriesSplit(
            n_splits=7,
            gap=30,
            max_train_size=5000,
            test_size=1000,
        )

        all_splits = list(ts_cv.split(X, y))

        return all_splits

    def create_model(self):

        from keras.metrics import mean_absolute_error
        from keras import activations

        models = [model for model in vars(self) if model.__contains__("model")]

        sq = Sequential()
        sq.add(LSTM(128, return_sequences=True, input_shape=(1, 44)))  
        sq.add(LSTM(64, return_sequences=True)) 
        sq.add(LSTM(64, return_sequences=True)) 
        sq.add(LSTM(64, return_sequences=True)) 
        sq.add(GRU(64, return_sequences=True))
        sq.add(GRU(64, return_sequences=True))
        sq.add(GRU(64))
        sq.add(Dense(32, activation=activations.relu, activity_regularizer="l1_l2"))
        sq.add(Dropout(0.5))
        sq.add(Dense(16, activation=activations.relu, activity_regularizer="l1_l2"))
        sq.add(Dropout(0.5))
        sq.add(Dense(1, activation=activations.linear, activity_regularizer="l1_l2"))
        sq.build(input_shape=(1, 44))
        sq.summary()
        sq.compile(loss='mean_absolute_error', optimizer='adam')

        for model in models: 
            self.__setattr__(model, sq)

    def fit_model(self, model, X_train, y_train, training_parameters={}):

        if "epochs" in training_parameters : epochs = training_parameters["epochs"] 
        else: epochs = training_parameters["epochs"] = 15

        if "batch_size" in training_parameters : batch_size = training_parameters["batch_size"] 
        else: batch_size = training_parameters["batch_size"] =  128

        model.fit(
            X_train,
            y_train,
            epochs,
            batch_size,
            verbose = 2,
        )

        return model
    
    def fit_model_timeseries(self, model, all_splits, X_train, y_train, training_parameters={}): 
        from tqdm import tqdm
        
        if "epochs" in training_parameters : epochs = training_parameters["epochs"] 
        else: epochs = training_parameters["epochs"] = 15

        if "batch_size" in training_parameters : batch_size = training_parameters["batch_size"] 
        else: batch_size = training_parameters["batch_size"] =  128
        
        for i in tqdm(range(0, epochs)): 
            for train, test in (all_splits): 
                X = np.array(X_train.iloc[train])
                X = np.reshape(X, ((X.shape[0], 1, X.shape[1])))
            
                X_val = np.array(X_train.iloc[test])
                X_val = np.reshape(X_val, ((X_val.shape[0], 1, X_val.shape[1])))
                
                
                model.fit(
                        X, 
                        y_train.iloc[train], 
                        validation_data=(X_val, y_train.iloc[test]),
                        shuffle=False,
                        epochs=1,
                        batch_size=batch_size,
                )
            



        


        
    

    
