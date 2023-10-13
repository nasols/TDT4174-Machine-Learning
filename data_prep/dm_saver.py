import pickle

class DM_Saver() : 

    def __init__(self) : 

        self.data = {}

    def set_data(self, dm): 
        self.data = {
            'train_a': dm.train_a,
            'train_b': dm.train_b,
            'train_c': dm.train_c,
            'X_train_estimated_a': dm.X_train_estimated_a,
            'X_train_estimated_b': dm.X_train_estimated_b,
            'X_train_estimated_c': dm.X_train_estimated_c,
            'X_train_observed_a': dm.X_train_observed_a,
            'X_train_observed_b': dm.X_train_observed_b,
            'X_train_observed_c': dm.X_train_observed_c,
            'X_test_estimated_a': dm.X_test_estimated_a,
            'X_test_estimated_b': dm.X_test_estimated_b,
            'X_test_estimated_c': dm.X_test_estimated_c,
            'data_A_obs': dm.data_A_obs,
            'data_B_obs': dm.data_B_obs,
            'data_C_obs': dm.data_C_obs,
            'data_A_es': dm.data_A_es,
            'data_B_es': dm.data_B_es,
            'data_C_es': dm.data_C_es,
            'amplitude': dm.amplitude
        }

    def save(self, filename):
        # Serialize and save the Data_Manager instance to a file
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        # Load a serialized Data_Manager instance from a file
        with open(filename, 'rb') as file:
            dms = pickle.load(file)

        return dms