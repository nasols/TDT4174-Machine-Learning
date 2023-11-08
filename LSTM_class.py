from keras import Sequential
import pandas as pd 
import numpy as np 
from keras.layers import LSTM, Dense, GRU, Dropout, Normalization, Bidirectional, TimeDistributed

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

    def create_model(self, num_input_features, timesteps):

        models = [model for model in vars(self) if model.__contains__("model")]

        sq = Sequential()
        # sq.add(LSTM(128, return_sequences=True, input_shape=(timesteps, num_input_features)))
        # sq.add(LSTM(128, return_sequences=True))
        # sq.add(LSTM(64, return_sequences=True))
        # sq.add(LSTM(64, return_sequences=True))
        # sq.add(Dense(128, activity_regularizer="l2")) # added l1 
        # sq.add(Dropout(0.5))
        # sq.add(Dense(64, activity_regularizer="l2")) # added l1
        # sq.add(Dropout(0.5))
        # sq.add(Dense(num_input_features, activity_regularizer="l2")) # added l1 
        # sq.add(Dense(1))

        sq.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(timesteps, num_input_features))))
        sq.add(Bidirectional(LSTM(64, return_sequences=True)))
        sq.add(Bidirectional(LSTM(32, return_sequences=True)))
        sq.add(TimeDistributed(Dense(64, activity_regularizer="l1")))
        sq.add(TimeDistributed(Dense(num_input_features, activity_regularizer="l1")))
        sq.add(TimeDistributed(Dense(1, activity_regularizer="l1")))

        # sq.add(Dense(2*256, activity_regularizer="l2"))
        # sq.add(Dense(256, activity_regularizer="l2"))
        # sq.add(Dense(256, activity_regularizer="l2"))
        # sq.add(Dense(128, activity_regularizer="l2"))
        # sq.add(Dense(128, activity_regularizer="l2"))
        # sq.add(Dense(64, activity_regularizer="l2"))
        # sq.add(Dense(64, activity_regularizer="l2"))
        # sq.add(Dense(num_input_features))
        # sq.add(Dense(1))

        sq.build(input_shape=(None, timesteps,num_input_features))
        sq.summary()

        sq.compile(loss='mean_absolute_error', optimizer='rmsprop') 

        for model in models: 
            self.__setattr__(model, sq)

    def fit_model(self, model, X_train, y_train, training_parameters={}, kfolds=False):
        from sklearn.model_selection import KFold

        if "epochs" in training_parameters : epochs = training_parameters["epochs"] 
        else: epochs = training_parameters["epochs"] = 15

        if "batch_size" in training_parameters : batch_size = training_parameters["batch_size"] 
        else: batch_size = training_parameters["batch_size"] =  128

        if "input_features" in training_parameters : input_features = training_parameters["input_features"]
        else: input_features = training_parameters["input_features"] = X_train.shape[1]

        if "timesteps" in training_parameters : timesteps = training_parameters["timesteps"]
        else: timesteps = training_parameters["timesteps"] = 1            

        if kfolds: 

            i = 0

            skf = KFold(n_splits=10, shuffle=False)

            while i <= epochs: 
                print(f"Epoch Number: {i}")
                for train_index, test_index in skf.split(X_train): 

                    i += 1

                    X, X_test = X_train.iloc[train_index], X_train.iloc[test_index]
                    y, y_test = y_train.iloc[train_index], y_train.iloc[test_index]

                    X_t = np.asarray(X).astype(float)
                    X_t = np.reshape(X_t, ((X_t.shape[0], 1, X_t.shape[1])))

                    X_val = np.asarray(X_test).astype(float)
                    X_val = np.reshape(X_val, ((X_val.shape[0], 1, X_val.shape[1])))


                    y_t = np.asarray(y).astype(float)

                    y_val = np.asarray(y_test).astype(float)

                    print(f"Fold {i%10}")
                    print(f"Train: index={train_index}")
                    print(f"Test: index={test_index}")

                    model.fit(X_t, 
                            y_t,
                            epochs=1,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val))

        else: 

            X = np.asarray(X_train).astype(float)
            X = np.reshape(X, (int(X.shape[0]/timesteps), timesteps, input_features))

            y = np.asarray(y_train).astype(float)
            y = y.reshape((int(y.shape[0]/timesteps),timesteps,))
            model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
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
                X = np.asarray(X_train.iloc[train]).astype(float)
                X = np.reshape(X, ((X.shape[0], 1, X.shape[1])))

                y = np.asarray(y_train.iloc[train]).astype(float)
            
                X_val = np.asarray(X_train.iloc[test]).astype(float)
                X_val = np.reshape(X_val, ((X_val.shape[0], 1, X_val.shape[1])))

                y_val = np.asarray(y_train.iloc[test]).astype(float)               
                
                model.fit(
                        X, 
                        y, 
                        validation_data=(X_val, y_val),
                        shuffle=False,
                        epochs=1,
                        batch_size=batch_size,
                        
                )

    def shape_datasets(self, num_features, timesteps, X_data, y_data):
        
        """ 
        know our test data is of shape (720, ) so our timesteps must divide 720 to shape (something, timesteps, None)
        need to shape X into (something, timesteps, num_features)
        need to find divisor common for something and 720 that is closest to timesteps 

        NB: timesteps should divide 720 

        """

        divisors_submission = self.divisorGenerator(720)

        assert timesteps in divisors_submission, "timesteps input should divide 720 (our submission dimention)"

        train_shape = X_data.shape[0]

        for i in range(100): 
            if ( train_shape % timesteps == 0) : 
                print(train_shape)
                break
            
            else: 
                train_shape += 1 

        needed_padding = train_shape - X_data.shape[0]

        padding_x = pd.DataFrame(np.zeros((needed_padding, num_features)))
        padding_x.columns = X_data.columns

        padding_y = pd.DataFrame(np.zeros((needed_padding, )))

        shaped_X = pd.concat([X_data, padding_x], ignore_index=True)
        shaped_y = pd.concat([y_data, padding_y], ignore_index=True)

        return shaped_X, shaped_y

    def predict_model(self, model, X, num_features, timesteps): 

        pass 

    def divisorGenerator(self, n):
        x = []

        for i in range(1, n//2+1):
            if n%i == 0: x.append(i)
        
        return x

    def closest(self, lst, K):
     
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
     

        


        
    

    
